# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/compiler target/TORQ)
# The torq_next plugin uses its own CMakeLists.txt with IREE_PACKAGE_ROOT_PREFIX
# set inside it, so no separate iree_setup_c_src_root is needed here.
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/compiler/torq_next target/torq_next)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/compiler/tools ${CMAKE_BINARY_DIR}/compiler/tools)
