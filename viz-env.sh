#!/bin/bash
# Ensure pkg-config can find .pc files from conda packages during PyPI builds
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
