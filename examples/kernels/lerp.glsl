#version 120
uniform sampler2D a;
uniform sampler2D b;
uniform float alpha;
varying vec2 x; // texture coordinates
uniform float dx; // delta

void main()
{
    // gl_FragData[0] is buffer out
    gl_FragData[0] = texture2D(a,x);
}