#version 460

layout (location = 0) in vec2 vp;
layout (location = 1) in vec2 vel;
layout (location = 2) in float density;

out vec4 colour;

void main() {
	gl_Position = vec4(vp, 1.0, 1.0);
	colour = vec4(vel.x, vel.y, 0.f , density);
};