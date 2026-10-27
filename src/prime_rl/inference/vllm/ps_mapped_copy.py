"""Coalesced copies through GPU execution units, leaving DMA engines available for decoding."""

import triton
import triton.language as tl


@triton.jit
def _mapped_rows(
    source_address,
    destination_address,
    source_pitch: tl.constexpr,
    destination_pitch: tl.constexpr,
    WIDTH: tl.constexpr,
    WORD: tl.constexpr,
    SOURCE_HOST: tl.constexpr,
    DESTINATION_HOST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    dtype = tl.uint64 if WORD == 8 else tl.uint32 if WORD == 4 else tl.uint16 if WORD == 2 else tl.uint8
    source, destination = (
        source_address.to(tl.pointer_type(dtype)),
        destination_address.to(tl.pointer_type(dtype)),
    )
    tiles = tl.cdiv(WIDTH, BLOCK)
    row, tile = tl.program_id(0) // tiles, tl.program_id(0) % tiles
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(
        source + row * source_pitch + cols,
        cols < WIDTH,
        0,
        cache_modifier=".cv" if SOURCE_HOST else ".ca",
    )
    tl.store(
        destination + row * destination_pitch + cols,
        value,
        cols < WIDTH,
        cache_modifier=".wt" if DESTINATION_HOST else ".wb",
    )


@triton.jit
def _mapped_rows_limited(
    source_address,
    destination_address,
    source_pitch: tl.constexpr,
    destination_pitch: tl.constexpr,
    WIDTH: tl.constexpr,
    WORD: tl.constexpr,
    SOURCE_HOST: tl.constexpr,
    DESTINATION_HOST: tl.constexpr,
    BLOCK: tl.constexpr,
    TILES,
):
    dtype = tl.uint64 if WORD == 8 else tl.uint32 if WORD == 4 else tl.uint16 if WORD == 2 else tl.uint8
    source = source_address.to(tl.pointer_type(dtype))
    destination = destination_address.to(tl.pointer_type(dtype))
    tiles = tl.cdiv(WIDTH, BLOCK)
    for index in range(tl.program_id(0), TILES, tl.num_programs(0)):
        row, tile = index // tiles, index % tiles
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(
            source + row * source_pitch + cols,
            cols < WIDTH,
            0,
            cache_modifier=".cv" if SOURCE_HOST else ".ca",
        )
        tl.store(
            destination + row * destination_pitch + cols,
            value,
            cols < WIDTH,
            cache_modifier=".wt" if DESTINATION_HOST else ".wb",
        )


def mapped_copy(destination, source, *, programs=0):
    size = source.element_size()
    width = source.shape[1] * size
    values = (
        source.data_ptr(),
        destination.data_ptr(),
        source.stride(0) * size,
        destination.stride(0) * size,
        width,
    )
    word = 8
    while any(value % word for value in values):
        word //= 2
    tiles = source.shape[0] * triton.cdiv(width // word, 512)
    kernel = _mapped_rows_limited if programs else _mapped_rows
    grid = min(tiles, programs) if programs else tiles
    extra = (tiles,) if programs else ()
    kernel[(grid,)](
        source.data_ptr(),
        destination.data_ptr(),
        source.stride(0) * size // word,
        destination.stride(0) * size // word,
        width // word,
        word,
        not source.is_cuda,
        not destination.is_cuda,
        512,
        *extra,
        num_warps=4,
    )
