#version 460
out vec4 FragColor;

in vec2 TexCoord;

// texture samplers
uniform sampler2D textureNodes;

void main()
{
	// linearly interpolate between both textures (80% container, 20% awesomeface)
	FragColor = texture(textureNodes, TexCoord);
}