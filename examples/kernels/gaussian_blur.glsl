#version 120
uniform sampler2D input;
uniform int n;
varying vec2 x; // texture coordinates
uniform float dx; // delta

void main()
{
    vec3 v = vec3(0,0,0);
    float totWeight = 0;
    for(int j = -n; j<=n; ++j) {
        for(int i = -n; i<=n; ++i) {
            vec2 tx = x + vec2(i*dx,j*dx);
            float weight = exp(-((tx.x-x.x)*(tx.x-x.x)+(tx.y-x.y)*(tx.y-x.y)));
            v+= weight * texture2D(input, tx).xyz;
            totWeight += weight;
        }
    }
    
    // gl_FragData[0] is buffer output
    gl_FragData[0] = vec4(v/totWeight,1);
}
