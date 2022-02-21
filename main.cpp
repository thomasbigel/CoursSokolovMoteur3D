#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include <iostream>

const TGAColor white = TGAColor(255, 255, 255);
const TGAColor red   = TGAColor(255, 0,   0);
const TGAColor green = TGAColor(0,   255, 0);
const TGAColor blue  = TGAColor(0,   0,   255);
const int width  = 800; 
const int height = 800; 
Model *model = NULL;
float *zbuffer = new float[width*height];

void line(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor &color) { 
    bool steep = false; 
    if (std::abs(x0-x1)<std::abs(y0-y1)) { 
        std::swap(x0, y0); 
        std::swap(x1, y1); 
        steep = true; 
    } 
    if (x0>x1) { 
        std::swap(x0, x1); 
        std::swap(y0, y1); 
    } 
    int dx = x1-x0; 
    int dy = y1-y0; 
    int derror2 = std::abs(dy)*2; 
    int error2 = 0; 
    int y = y0; 
    for (int x=x0; x<=x1; x++) { 
        if (steep) { 
            image.set(y, x, color); 
        } else { 
            image.set(x, y, color); 
        } 
        error2 += derror2; 
        if (error2 > dx) { 
            y += (y1>y0?1:-1); 
            error2 -= dx*2; 
        } 
    } 
} 

void lineVec(Vec2i p0, Vec2i p1, TGAImage &image, TGAColor color) { 
    //line(p0->x, p0->y, p1->x, p1->y, image, color);
    line(p0.x, p0.y, p1.x, p1.y, image, color);
}

Vec3f barycentric(Vec2i *pts, Vec3f P) { 
    Vec3f u = crossGeo(Vec3f(pts[2][0]-pts[0][0], pts[1][0]-pts[0][0], pts[0][0]-P[0]), Vec3f(pts[2][1]-pts[0][1], pts[1][1]-pts[0][1], pts[0][1]-P[1]));
    /* `pts` and `P` has integer value as coordinates
       so `abs(u[2])` < 1 means `u[2]` is 0, that means
       triangle is degenerate, in this case return something with negative coordinates */
    if (std::abs(u[2])<1) return Vec3f(-1,1,1);
    return Vec3f(1.f-(u.x+u.y)/u.z, u.y/u.z, u.x/u.z); 
} 

Vec3f barycentric(Vec3f *pts, Vec3f P) { 
    Vec3f u = crossGeo(Vec3f(pts[2][0]-pts[0][0], pts[1][0]-pts[0][0], pts[0][0]-P[0]), Vec3f(pts[2][1]-pts[0][1], pts[1][1]-pts[0][1], pts[0][1]-P[1]));
    /* `pts` and `P` has integer value as coordinates
       so `abs(u[2])` < 1 means `u[2]` is 0, that means
       triangle is degenerate, in this case return something with negative coordinates */
    if (std::abs(u[2])<1) return Vec3f(-1,1,1);
    return Vec3f(1.f-(u.x+u.y)/u.z, u.y/u.z, u.x/u.z); 
} 

void triangle(Vec3f *pts, TGAImage &image, float intensity, Vec2i *uv) {
    Vec2f bboxmin(image.get_width()-1,  image.get_height()-1); 
    Vec2f bboxmax(0, 0); 
    Vec2f clamp(image.get_width()-1, image.get_height()-1); 
    for (int i=0; i<3; i++) { 
        for (int j=0; j<2; j++) { 
            bboxmin[j] = std::max(0.f,        std::min(bboxmin[j], pts[i][j])); 
            bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], pts[i][j])); 
        } 
    } 
    
    //float xBboxmin = bboxmin.x;
    int xBboxmax = bboxmax.x;

    //loop sur les points à l'interieur de la box 2D du triangle
    #pragma omp parallel for
    for (int x=bboxmin.x; x<=xBboxmax; x++) { 
        //recree variable pour parallel
        Vec3f P; 
        P.x = x;
        int yBboxmax = bboxmax.y;
        #pragma omp parallel for
        for (int y=bboxmin.y; y<=yBboxmax; y++) {
            //recree variable pour parallel
            Vec2i uvP;
            Vec3f P; 
            P.x = x;
            P.y = y;


            Vec3f bary_point = barycentric(pts, P);
            if (bary_point.x<0 || bary_point.y<0 || bary_point.z<0) continue; 
            P.z = 0;
            for(int i=0; i<3; i++){
                P.z += pts[i][2] * bary_point[i];
            }
            uvP.x = bary_point.x * uv[0].x + bary_point.y * uv[1].x + bary_point.z * uv[2].x;
            uvP.y = bary_point.x * uv[0].y + bary_point.y * uv[1].y + bary_point.z * uv[2].y;
            if(zbuffer[int(P.x + P.y * width)] < P.z){
                zbuffer[int(P.x + P.y * width)] = P.z;
                TGAColor color = model->diffuse(uvP);
                image.set(P.x, P.y, TGAColor(color.r*intensity, color.g*intensity, color.b*intensity, 255));
            }

        }
    }
}

void rasterize(Vec2i p0, Vec2i p1, TGAImage &image, TGAColor color, int ybuffer[]) {
    if (p0.x>p1.x) {
        std::swap(p0, p1);
    }
    for (int x=p0.x; x<=p1.x; x++) {
        float t = (x-p0.x)/(float)(p1.x-p0.x);
        int y = p0.y*(1.-t) + p1.y*t;
        if (ybuffer[x]<y) {
            ybuffer[x] = y;
            image.set(x, 0, color);
        }
    }
}

int main(int argc, char** argv) {
    if (2==argc) {
        model = new Model(argv[1]);
    } else {
        model = new Model("obj/african_head.obj");
    }

    for (int i=width*height; i--; zbuffer[i] = -std::numeric_limits<float>::max());

    TGAImage image(width, height, TGAImage::RGB);
    Vec3f light_dir(0,0,-1);
    for (int i=0; i<model->nfaces(); i++) {
        std::vector<int> face = model->face(i);
        Vec3f screen_coords[3];
        Vec3f world_coords[3];
        for (int j=0; j<3; j++) {
            Vec3f v = model->vert(face[j]);
            screen_coords[j] = Vec3f(int((v.x+1.)*width/2.+.5), int((v.y+1.)*height/2.+.5), v.z);
            world_coords[j]  = v;
        }
        Vec3f n = (world_coords[2]-world_coords[0])^(world_coords[1]-world_coords[0]);
        n.normalize();
        float intensity = n*light_dir;
        if (intensity>0) {
            Vec2i uv[3];
            for (int k=0; k<3; k++) {
                uv[k] = model->uv(i, k);
            }
            triangle(screen_coords, image, intensity, uv);
        }
    }

    image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
    image.write_tga_file("output.tga");
    delete model;

/* 2D
// just dumping the 2d scene (yay we have enough dimensions!)
    TGAImage scene(width, height, TGAImage::RGB);

    // scene "2d mesh"
    lineVec(Vec2i(20, 34),   Vec2i(744, 400), scene, red);
    lineVec(Vec2i(120, 434), Vec2i(444, 400), scene, green);
    lineVec(Vec2i(330, 463), Vec2i(594, 200), scene, blue);

    // screen line
    lineVec(Vec2i(10, 10), Vec2i(790, 10), scene, white);

    scene.flip_vertically(); // i want to have the origin at the left bottom corner of the image
    scene.write_tga_file("scene.tga");

    TGAImage render(width, 16, TGAImage::RGB);
    int ybuffer[width];
    for (int i=0; i<width; i++) {
         ybuffer[i] = std::numeric_limits<int>::min();
     }
    rasterize(Vec2i(20, 34),   Vec2i(744, 400), render, red,   ybuffer);
    rasterize(Vec2i(120, 434), Vec2i(444, 400), render, green, ybuffer);
    rasterize(Vec2i(330, 463), Vec2i(594, 200), render, blue,  ybuffer);

    // 1-pixel wide image is bad for eyes, lets widen it
    for (int i=0; i<width; i++) {
        for (int j=1; j<16; j++) {
            render.set(i, j, render.get(i, 0));
        }
    }
    render.flip_vertically(); // i want to have the origin at the left bottom corner of the image
    render.write_tga_file("render.tga");
*/


    return 0;
}
