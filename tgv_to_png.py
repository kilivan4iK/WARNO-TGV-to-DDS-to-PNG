import argparse
import io
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import zstandard as zstd

try:
    import numpy as np
except ImportError:  # optional, used only for BC5 normal-Z reconstruction
    np = None


TABLE_CANDIDATES = (0x30, 0x34, 0x38, 0x3C)


@dataclass(frozen=True)
class TGVInfo:
    path: Path
    version: int
    unk: int
    width: int
    height: int
    mip_count: int
    fmt: str
    data: bytes
    table_start: int
    offsets: list[int]
    sizes: list[int]


def normalize_format(raw_fmt: bytes) -> str:
    text = raw_fmt.decode("ascii", errors="ignore").upper()
    patterns = (
        r"BC[1-7](?:_[A-Z0-9]+)?",
        r"A8B8G8R8(?:_[A-Z0-9]+)?",
        r"L16(?:_[A-Z0-9]+)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return text.strip() or "UNKNOWN"


def try_table(data: bytes, table_start: int, mip_count: int) -> tuple[int, list[int] | None, list[int] | None]:
    try:
        offsets = list(struct.unpack_from(f"<{mip_count}I", data, table_start))
        sizes = list(struct.unpack_from(f"<{mip_count}I", data, table_start + 4 * mip_count))
    except struct.error:
        return 0, None, None

    valid = 0
    for offset, size in zip(offsets, sizes):
        if (
            0 <= offset <= len(data) - 12
            and 12 <= size <= len(data)
            and offset + size <= len(data)
            and data[offset : offset + 4] == b"ZSTD"
        ):
            valid += 1
    return valid, offsets, sizes


def parse_tgv(path: Path) -> TGVInfo:
    data = path.read_bytes()
    if len(data) < 0x2C:
        raise RuntimeError(f"{path.name}: file too small to be valid TGV")

    version, unk, width, height = struct.unpack_from("<4I", data, 0)
    mip_count = struct.unpack_from("<H", data, 0x18)[0]
    fmt = normalize_format(data[0x1C : 0x1C + 16])

    best = (0, None, None, None)
    for table_start in TABLE_CANDIDATES:
        valid, offsets, sizes = try_table(data, table_start, mip_count)
        if valid > best[0]:
            best = (valid, table_start, offsets, sizes)

    valid, table_start, offsets, sizes = best
    if valid == 0 or table_start is None or offsets is None or sizes is None:
        raise RuntimeError(f"{path.name}: could not find valid mip offset/size table")

    return TGVInfo(
        path=path,
        version=version,
        unk=unk,
        width=width,
        height=height,
        mip_count=mip_count,
        fmt=fmt,
        data=data,
        table_start=table_start,
        offsets=offsets,
        sizes=sizes,
    )


def iter_valid_mips(info: TGVInfo) -> Iterable[tuple[int, int, int, int]]:
    for idx, (offset, size) in enumerate(zip(info.offsets, info.sizes)):
        if (
            0 <= offset <= len(info.data) - 12
            and 12 <= size <= len(info.data)
            and offset + size <= len(info.data)
            and info.data[offset : offset + 4] == b"ZSTD"
        ):
            raw_size = struct.unpack_from("<I", info.data, offset + 4)[0]
            yield idx, offset, size, raw_size


def decompress_mip(info: TGVInfo, offset: int, size: int, raw_size: int) -> bytes:
    comp = info.data[offset + 8 : offset + size]
    reader = zstd.ZstdDecompressor().stream_reader(io.BytesIO(comp))
    try:
        raw = reader.read(raw_size)
    finally:
        reader.close()

    if len(raw) != raw_size:
        raise RuntimeError(
            f"{info.path.name}: decompression size mismatch at 0x{offset:X}: "
            f"got {len(raw)}, expected {raw_size}"
        )
    return raw


def expected_fullres_size(width: int, height: int, fmt: str) -> int | None:
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4

    if "BC1" in fmt:
        return blocks_x * blocks_y * 8
    if "BC3" in fmt or "BC5" in fmt or "BC7" in fmt:
        return blocks_x * blocks_y * 16
    if "A8B8G8R8" in fmt:
        return width * height * 4
    if "L16" in fmt:
        return width * height * 2
    return None


def pick_fullres_mip(info: TGVInfo) -> tuple[int, int, int, int]:
    mips = list(iter_valid_mips(info))
    if not mips:
        raise RuntimeError(f"{info.path.name}: no valid ZSTD mip entries found")

    expected = expected_fullres_size(info.width, info.height, info.fmt)
    if expected is not None:
        for mip in reversed(mips):
            if mip[3] == expected:
                return mip

    return max(mips, key=lambda x: x[3])


def build_dds_header_compressed(
    width: int,
    height: int,
    top_linear_size: int,
    fourcc: bytes,
    dxgi_format: int | None = None,
) -> bytes:
    dds_magic = b"DDS "
    dds_header_size = 124
    dds_pf_size = 32

    ddsd_caps = 0x1
    ddsd_height = 0x2
    ddsd_width = 0x4
    ddsd_pixel_format = 0x1000
    ddsd_linear_size = 0x80000

    ddpf_fourcc = 0x4
    ddscaps_texture = 0x1000

    flags = ddsd_caps | ddsd_height | ddsd_width | ddsd_pixel_format | ddsd_linear_size

    header = struct.pack("<I", dds_header_size)
    header += struct.pack("<I", flags)
    header += struct.pack("<I", height)
    header += struct.pack("<I", width)
    header += struct.pack("<I", top_linear_size)
    header += struct.pack("<I", 0)
    header += struct.pack("<I", 1)
    header += struct.pack("<11I", *([0] * 11))

    pf = struct.pack("<I", dds_pf_size)
    pf += struct.pack("<I", ddpf_fourcc)
    pf += fourcc
    pf += struct.pack("<I", 0)
    pf += struct.pack("<I", 0)
    pf += struct.pack("<I", 0)
    pf += struct.pack("<I", 0)
    pf += struct.pack("<I", 0)
    header += pf

    header += struct.pack("<I", ddscaps_texture)
    header += struct.pack("<I", 0)
    header += struct.pack("<I", 0)
    header += struct.pack("<I", 0)
    header += struct.pack("<I", 0)

    out = dds_magic + header
    if dxgi_format is not None:
        out += struct.pack("<5I", dxgi_format, 3, 0, 1, 0)
    return out


def decode_block_compressed(raw: bytes, width: int, height: int, fmt: str) -> Image.Image:
    fmt_up = fmt.upper()

    if "BC1" in fmt_up:
        fourcc = b"DXT1"
        dxgi = None
    elif "BC3" in fmt_up:
        fourcc = b"DXT5"
        dxgi = None
    elif "BC5" in fmt_up:
        fourcc = b"ATI2"
        dxgi = None
    elif "BC7" in fmt_up:
        fourcc = b"DX10"
        dxgi = 99 if "SRGB" in fmt_up else 98  # BC7_UNORM_SRGB / BC7_UNORM
    else:
        raise RuntimeError(f"Unsupported compressed format: {fmt}")

    dds_blob = build_dds_header_compressed(width, height, len(raw), fourcc, dxgi) + raw
    image = Image.open(io.BytesIO(dds_blob))
    image.load()

    if "BC5" in fmt_up:
        return image.convert("RGB")
    return image.convert("RGBA")


def decode_uncompressed(raw: bytes, width: int, height: int, fmt: str) -> Image.Image:
    fmt_up = fmt.upper()

    if "A8B8G8R8" in fmt_up:
        expected = width * height * 4
        if len(raw) != expected:
            raise RuntimeError(f"A8B8G8R8 size mismatch: got {len(raw)}, expected {expected}")
        return Image.frombytes("RGBA", (width, height), raw)

    if "L16" in fmt_up:
        expected = width * height * 2
        if len(raw) != expected:
            raise RuntimeError(f"L16 size mismatch: got {len(raw)}, expected {expected}")
        return Image.frombytes("I;16", (width, height), raw)

    raise RuntimeError(f"Unsupported uncompressed format: {fmt}")


def decode_tgv_image(info: TGVInfo, raw: bytes) -> Image.Image:
    fmt_up = info.fmt.upper()
    if "BC" in fmt_up:
        return decode_block_compressed(raw, info.width, info.height, info.fmt)
    return decode_uncompressed(raw, info.width, info.height, info.fmt)


def detect_texture_role(path: Path, fmt: str) -> str:
    name = path.stem.lower()
    fmt_up = fmt.upper()

    if "combinedda" in name or "coloralpha" in name:
        return "combined_da"
    if "normal" in name or "tscnm" in name or "BC5" in fmt_up:
        return "normal"
    if "combinedorm" in name or "_orm" in name or "ormtexture" in name:
        return "orm"
    if "splat" in name:
        return "splat"
    if "height" in name or "L16" in fmt_up:
        return "height"
    return "generic"


def normal_reconstruct_z(rgb: Image.Image) -> tuple[Image.Image, Image.Image]:
    if np is None:
        raise RuntimeError("NumPy is not installed, cannot reconstruct BC5 normal Z channel")

    arr = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    x = arr[:, :, 0] / 255.0 * 2.0 - 1.0
    y = arr[:, :, 1] / 255.0 * 2.0 - 1.0
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))

    out = np.empty_like(arr, dtype=np.uint8)
    out[:, :, 0] = np.clip(arr[:, :, 0], 0, 255).astype(np.uint8)
    out[:, :, 1] = np.clip(arr[:, :, 1], 0, 255).astype(np.uint8)
    out[:, :, 2] = np.clip((z * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    z_gray = np.clip((z * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB"), Image.fromarray(z_gray, mode="L")


def preview_8bit_from_16bit(image: Image.Image) -> Image.Image:
    raw = image.tobytes()
    if image.mode == "I;16B":
        high_bytes = raw[0::2]
    else:
        high_bytes = raw[1::2]
    return Image.frombytes("L", image.size, high_bytes)


def save_auto_channels(image: Image.Image, role: str, out_main: Path) -> list[Path]:
    out_paths: list[Path] = []
    stem = out_main.with_suffix("")

    if role == "orm":
        rgb = image.convert("RGB")
        for channel, suffix in zip(rgb.split(), ("occlusion", "roughness", "metallic")):
            path = stem.with_name(f"{stem.name}_{suffix}.png")
            channel.save(path)
            out_paths.append(path)

    elif role == "combined_da":
        rgba = image.convert("RGBA")
        r, g, b, a = rgba.split()

        diffuse_path = stem.with_name(f"{stem.name}_diffuse.png")
        alpha_path = stem.with_name(f"{stem.name}_alpha.png")

        Image.merge("RGB", (r, g, b)).save(diffuse_path)
        a.save(alpha_path)

        out_paths.extend((diffuse_path, alpha_path))

    elif role == "splat":
        rgba = image.convert("RGBA")
        for channel, suffix in zip(rgba.split(), ("mask_r", "mask_g", "mask_b", "mask_a")):
            path = stem.with_name(f"{stem.name}_{suffix}.png")
            channel.save(path)
            out_paths.append(path)

    elif role == "normal":
        rgb = image.convert("RGB")
        x_chan, y_chan, _ = rgb.split()

        x_path = stem.with_name(f"{stem.name}_normal_x.png")
        y_path = stem.with_name(f"{stem.name}_normal_y.png")
        x_chan.save(x_path)
        y_chan.save(y_path)
        out_paths.extend((x_path, y_path))

        if np is not None:
            reconstructed, z_chan = normal_reconstruct_z(rgb)
            recon_path = stem.with_name(f"{stem.name}_normal_reconstructed.png")
            z_path = stem.with_name(f"{stem.name}_normal_z.png")
            reconstructed.save(recon_path)
            z_chan.save(z_path)
            out_paths.extend((recon_path, z_path))

    elif role == "height" and image.mode in ("I;16", "I;16L", "I;16B"):
        preview = preview_8bit_from_16bit(image)
        preview_path = stem.with_name(f"{stem.name}_height_preview_8bit.png")
        preview.save(preview_path)
        out_paths.append(preview_path)

    return out_paths


def save_all_channels(image: Image.Image, out_main: Path) -> list[Path]:
    out_paths: list[Path] = []
    stem = out_main.with_suffix("")

    if image.mode in ("RGB", "RGBA"):
        labels = ("r", "g", "b", "a")
        for idx, channel in enumerate(image.split()):
            path = stem.with_name(f"{stem.name}_{labels[idx]}.png")
            channel.save(path)
            out_paths.append(path)
    elif image.mode in ("I;16", "I;16L", "I;16B"):
        preview = preview_8bit_from_16bit(image)
        path = stem.with_name(f"{stem.name}_8bit.png")
        preview.save(path)
        out_paths.append(path)
    return out_paths


def resolve_output_file(in_file: Path, output_arg: str | None) -> Path:
    if output_arg is None:
        return in_file.with_suffix(".png")

    out = Path(output_arg)
    if out.suffix.lower() == ".png":
        return out
    return out / f"{in_file.stem}.png"


def convert_one(in_file: Path, out_file: Path, split_mode: str) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    info = parse_tgv(in_file)
    mip_idx, offset, size, raw_size = pick_fullres_mip(info)
    raw = decompress_mip(info, offset, size, raw_size)
    image = decode_tgv_image(info, raw)

    image.save(out_file)
    role = detect_texture_role(in_file, info.fmt)

    extras: list[Path] = []
    if split_mode == "auto":
        extras = save_auto_channels(image, role, out_file)
    elif split_mode == "all":
        extras = save_all_channels(image, out_file)

    print(
        f"[OK] {in_file.name} -> {out_file.name} | "
        f"fmt={info.fmt} size={info.width}x{info.height} fullMip={mip_idx}/{info.mip_count - 1} role={role}"
    )
    for extra in extras:
        print(f"     + {extra.name}")


def convert_path(input_path: Path, output_arg: str | None, recursive: bool, split_mode: str) -> None:
    if input_path.is_file():
        convert_one(input_path, resolve_output_file(input_path, output_arg), split_mode)
        return

    if not input_path.is_dir():
        raise RuntimeError(f"Input path does not exist: {input_path}")

    out_dir = Path(output_arg) if output_arg else input_path / "png_out"
    pattern = "**/*.tgv" if recursive else "*.tgv"
    files = sorted(input_path.glob(pattern))

    if not files:
        print(f"No .tgv files found in {input_path}")
        return

    for file_path in files:
        rel = file_path.relative_to(input_path)
        out_file = (out_dir / rel).with_suffix(".png")
        convert_one(file_path, out_file, split_mode)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert WARNO TGV textures directly to PNG, with optional channel extraction."
    )
    parser.add_argument("input", help="Input .tgv file or folder with .tgv files")
    parser.add_argument("output", nargs="?", help="Output .png path (for single file) or output folder")
    parser.add_argument(
        "--split",
        choices=("auto", "all", "none"),
        default="auto",
        help="Channel extraction mode (default: auto)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If input is a folder, search .tgv files recursively",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        convert_path(Path(args.input), args.output, args.recursive, args.split)
    except Exception as exc:  # keep CLI output user-friendly
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

