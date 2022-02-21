#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "our_gl.h"
#include <iostream>
#include <string>

const TGAColor white = TGAColor(255, 255, 255);
const TGAColor red   = TGAColor(255, 0,   0);
const TGAColor green = TGAColor(0,   255, 0);
const TGAColor blue  = TGAColor(0,   0,   255);
const int width  = 800; 
const int height = 800; 
const float depth  = 2000.f;
Model *model = NULL;
float *zbuffer = NULL;
float *shadowbuffer = NULL;

Vec3f light_dir(1,1,0);
Vec3f       eye(1,1,4);
Vec3f    center(0,0,0);
Vec3f        up(0,1,0);

struct Shader : public IShader {
    mat<4,4,float> uniform_M;   //  Projection*ModelView
    mat<4,4,float> uniform_MIT; // (Projection*ModelView).invert_transpose()
    mat<4,4,float> uniform_Mshadow; // transform framebuffer screen coordinates to shadowbuffer screen coordinates
    mat<2,3,float> varying_uv;  // triangle uv coordinates, written by the vertex shader, read by the fragment shader
    mat<3,3,float> varying_tri; // triangle coordinates before Viewport transform, written by VS, read by FS

    Shader(Matrix M, Matrix MIT, Matrix MS) : uniform_M(M), uniform_MIT(MIT), uniform_Mshadow(MS), varying_uv(), varying_tri() {}

    virtual Vec4f vertex(int iface, int nthvert) {
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        Vec4f gl_Vertex = Viewport*Projection*ModelView*embed<4>(model->vert(iface, nthvert));
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex/gl_Vertex[3]));
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor &color) {
        Vec4f sb_p = uniform_Mshadow*embed<4>(varying_tri*bar); // corresponding point in the shadow buffer
        sb_p = sb_p/sb_p[3];
        int idx = int(sb_p[0]) + int(sb_p[1])*width; // index in the shadowbuffer array
        float shadow = .3+.7*(shadowbuffer[idx]<sb_p[2]+43.34); // magic coeff to avoid z-fighting
        Vec2f uv = varying_uv*bar;                 // interpolate uv for the current pixel
        Vec3f n = proj<3>(uniform_MIT*embed<4>(model->normal(uv))).normalize(); // normal
        Vec3f l = proj<3>(uniform_M  *embed<4>(light_dir        )).normalize(); // light vector
        Vec3f r = (n*(n*l*2.f) - l).normalize();   // reflected light
        float spec = pow(std::max(r.z, 0.0f), model->specular(uv));
        float diff = std::max(0.f, n*l);
        TGAColor c = model->diffuse(uv);
        for (int i=0; i<3; i++) color[i] = std::min<float>(20 + c[i]*shadow*(1.2*diff + .6*spec), 255);
        return false;
    }
};

struct DepthShader : public IShader {
    mat<3,3,float> varying_tri;

    DepthShader() : varying_tri() {}

    virtual Vec4f vertex(int iface, int nthvert) {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert)); // read the vertex from .obj file
        gl_Vertex = Viewport*Projection*ModelView*gl_Vertex;          // transform it to screen coordinates
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex/gl_Vertex[3]));
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor &color) {
        Vec3f p = varying_tri*bar;
        color = TGAColor(255, 255, 255)*(p.z/depth);
        return false;
    }
};

int main(int argc, char** argv) {

    if (2>argc) {
        std::cerr << "Usage: " << argv[0] << " obj/model.obj" << std::endl;
        return 1;
    }

    zbuffer = new float[width*height];
    shadowbuffer   = new float[width*height];
    for (int i=width*height; i--; zbuffer[i] = shadowbuffer[i] = -std::numeric_limits<float>::max());

    light_dir.normalize();

    TGAImage imageOutput(width, height, TGAImage::RGB);

    for (int m=1; m<argc; m++) {
        std::string nom("frame");
        std::string nomShadow("depth");
        std::string ms= std::to_string(m);
        std::string extensionImage(".tga");
        model = new Model(argv[m]);

        //Shadow buffer
        TGAImage depthImage(width, height, TGAImage::RGB);
        lookat(light_dir, center, up);
        viewport(width/8, height/8, width*3/4, height*3/4, depth);
        projection(0);
        DepthShader depthshader;
        Vec4f screen_coords_shadow[3];
        for (int i=0; i<model->nfaces(); i++) {
            for (int j=0; j<3; j++) {
                screen_coords_shadow[j] = depthshader.vertex(i, j);
            }
            triangle(screen_coords_shadow, depthshader, depthImage, shadowbuffer);
        }
        depthImage.flip_vertically(); // to place the origin in the bottom left corner of the image
        depthImage.write_tga_file((nomShadow + ms + extensionImage).c_str());

        Matrix M = Viewport*Projection*ModelView;
        
        //Shader
        TGAImage frame(width, height, TGAImage::RGB);
        lookat(eye, center, up);
        viewport(width/8, height/8, width*3/4, height*3/4, depth);
        projection(-1.f/(eye-center).norm());
        Shader shader(ModelView, (Projection*ModelView).invert_transpose(), M*(Viewport*Projection*ModelView).invert());
        Vec4f screen_coords[3];
        for (int i=0; i<model->nfaces(); i++) {
            for (int j=0; j<3; j++) {
                screen_coords[j] = shader.vertex(i, j);
            }
            triangle(screen_coords, shader, frame, zbuffer);
            triangle(screen_coords, shader, imageOutput, zbuffer);
        }
        delete model;
        frame.flip_vertically(); // to place the origin in the bottom left corner of the image
        frame.write_tga_file((nom + ms + extensionImage).c_str());
    }
    imageOutput.flip_vertically(); // to place the origin in the bottom left corner of the image
    imageOutput.write_tga_file("output.tga");

    delete []zbuffer;
    delete []shadowbuffer;

    return 0;
}
