#version 460

layout (location = 0) in vec2 vp;
layout (location = 1) in vec3 col;

out vec4 colour;

void main() {
	gl_Position = vec4(vp, 1.0, 1.0);
	colour = vec4(col, 1.0);
};