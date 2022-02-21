#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "our_gl.h"
#include <iostream>

const TGAColor white = TGAColor(255, 255, 255);
const TGAColor red   = TGAColor(255, 0,   0);
const TGAColor green = TGAColor(0,   255, 0);
const TGAColor blue  = TGAColor(0,   0,   255);
const int width  = 800; 
const int height = 800; 
const int depth  = 255;
Model *model = NULL;
float *zbuffer = new float[width*height];

Vec3f light_dir(1,1,1);
Vec3f       eye(1,1,3);
Vec3f    center(0,0,0);
Vec3f        up(0,1,0);

struct Shader : public IShader {
    mat<2,3,float> varying_uv;        // same as above
    mat<4,4,float> uniform_M;   //  Projection*ModelView
    mat<4,4,float> uniform_MIT; // (Projection*ModelView).invert_transpose()


    virtual Vec4f vertex(int iface, int nthvert) {
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert)); // read the vertex from .obj file
        return Viewport*Projection*ModelView*gl_Vertex; // transform it to screen coordinates
    }

    virtual bool fragment(Vec3f bar, TGAColor &color) {
        Vec2f uv = varying_uv*bar;
        Vec3f n = proj<3>(uniform_MIT*embed<4>(model->normal(uv))).normalize();
        Vec3f l = proj<3>(uniform_M  *embed<4>(light_dir        )).normalize();
        Vec3f r = (n*(n*l*2.f) - l).normalize();   // reflected light
        float spec = pow(std::max(r.z, 0.0f), model->specular(uv));
        float diff = std::max(0.f, n*l);
        TGAColor c = model->diffuse(uv);
        color = c;
        for (int i=0; i<3; i++) color[i] = std::min<float>(5 + c[i]*(diff + .6*spec), 255);
        return false;
    }
};

int main(int argc, char** argv) {
    if (2==argc) {
        model = new Model(argv[1]);
    } else {
        model = new Model("obj/african_head/african_head.obj");
        //model = new Model("obj/diablo3_pose/diablo3_pose.obj");
    }

    for (int i=width*height; i--; zbuffer[i] = -std::numeric_limits<float>::max());

    lookat(eye, center, Vec3f(0,1,0));
    viewport(width/8, height/8, width*3/4, height*3/4, depth);
    projection(-1.f/(eye-center).norm());
    light_dir.normalize();

    TGAImage image  (width, height, TGAImage::RGB);
    TGAImage zbuffer(width, height, TGAImage::GRAYSCALE);

    Shader shader;
    shader.uniform_M   =  Projection*ModelView;
    shader.uniform_MIT = (Projection*ModelView).invert_transpose();
    for (int i=0; i<model->nfaces(); i++) {
        Vec4f screen_coords[3];
        for (int j=0; j<3; j++) {
            //screen_coords[j] = Vec3f(int((v.x+1.)*width/2.+.5), int((v.y+1.)*height/2.+.5), v.z);
            //screen_coords[j] =  Vec3f(ViewPort*Projection*ModelView*Matrix(v));
            screen_coords[j] = shader.vertex(i, j);
        }
        triangle(screen_coords, shader, image, zbuffer);
    }

    image.  flip_vertically(); // to place the origin in the bottom left corner of the image
    zbuffer.flip_vertically();
    image.  write_tga_file("output.tga");
    zbuffer.write_tga_file("zbuffer.tga");

    delete model;
    return 0;


    return 0;
}
