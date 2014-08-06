#version 120
uniform sampler2D a;
uniform sampler2D b;
uniform float alpha;
varying vec2 x; // texture coordinates
uniform float dx; // delta

void main()
{
    // gl_FragData[0] is buffer out
    vec4 c = (1-alpha)*texture2D(a,x) + alpha*texture2D(b,x);
    c.w = 1;
    gl_FragData[0] = c;
}