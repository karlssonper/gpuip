#version 120
uniform sampler2D input;
uniform int n;
varying vec2 x; // texture coordinates
uniform float dx; // delta

void main()
{
    vec3 v = vec3(0,0,0);
    float count;
    for(int j = -n; j<=n; ++j) {
        for(int i = -n; i<=n; ++i) {
            vec2 tx = x + vec2(i*dx,j*dx);
            v+= texture2D(input, tx).xyz;
            count += 1;
        }
    }

    // gl_FragData[0] is buffer output
    gl_FragData[0] = vec4(v/count,1);
}
