#!/bin/bash -x

#set correct glslc location
GLSLC=glslc

$GLSLC vulkan_shader.vert -o vert.spv
$GLSLC vulkan_shader.frag -o frag.spv

cd ../../..

make vulkan_shaders