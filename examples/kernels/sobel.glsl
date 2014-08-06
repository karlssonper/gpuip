#version 120
uniform sampler2D input;
uniform float primary_treshold;
uniform float secondary_treshold;
varying vec2 x; // texture coordinates
uniform float dx; // delta

vec4 compute_sobelx()
{
    return -2*texture2D(input, x + vec2(-dx,0))
            - texture2D(input, x + vec2(-dx,dx))
            - texture2D(input, x + vec2(-dx,-dx))
            + 2*texture2D(input, x + vec2(dx,0))
            + texture2D(input, x + vec2(dx,dx))
            + texture2D(input, x + vec2(dx,-dx));
}

vec4 compute_sobely()
{
    return -2*texture2D(input, x + vec2(-dx,-dx))
            - texture2D(input, x + vec2(0,-dx))
            - texture2D(input, x + vec2(dx,-dx))
            + 2*texture2D(input, x + vec2(-dx,dx))
            + texture2D(input, x + vec2(0,dx))
            + texture2D(input, x + vec2(dx,dx));
}

float edge(float grad)
{
    if (grad > primary_treshold) {
        return 1.0;
    } else if (grad > secondary_treshold){
        return 0.5;
    } else {
        return 0.0;
    }
}

void main()
{
    vec4 sx = compute_sobelx();
    vec4 sy = compute_sobely();
    float grad = (abs(sx.x)+abs(sx.y)+abs(sx.z)+
                  abs(sx.x)+abs(sy.y)+abs(sy.z))/3.0;
    
    // gl_FragData[0] is buffer sobelx
    gl_FragData[0] = vec4(sx.xyz,1);

    // gl_FragData[1] is buffer sobely
    gl_FragData[1] = vec4(sy.xyz,1);

    // gl_FragData[2] is buffer gradient
    gl_FragData[2] = vec4(vec3(grad),1);

    // gl_FragData[3] is buffer edges
    gl_FragData[3] = vec4(vec3(edge(grad)),1);
}
