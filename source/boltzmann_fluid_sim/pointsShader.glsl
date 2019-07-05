#version 460

layout (location = 0) in vec2 vp;
layout (location = 1) in vec2 vel;
layout (location = 2) in float density;

out vec4 colour;

void colourWheel(in vec2 vel, out vec3 mappedColour) {
	vec2 R = { cos(1.0472), sin(1.0472) };
	vec2 G = { 1, 0 };
	vec2 B = { 0, 0 };

	vec3 r = { 255, 0, 0 };
	vec3 g = { 0, 255, 0 };
	vec3 b = { 0, 0, 255 };

	vec2 P = vel.y * R + vel.x * G + B * (1 - vel.x - vel.y);

	float PR = distance(P, R);
	float PG = distance(P, G);
	float PB = distance(P, B);

	mappedColour = normalize(PR * r + PG * g + PB * b);

}

void main() {
	gl_PointSize = 1;
	gl_Position = vec4(vp, 1.0, 1.0);

	vec3 wheelColour;
	colourWheel(vel, wheelColour);

	colour = vec4(wheelColour, density);
};