jet_positions[] = {1.5707963267948966, 4.71238898038469};
DefineConstant[
length = {2.2, Name "Channel length"}
front_distance = {0.2, Name "Cylinder center distance to inlet"}
bottom_distance = {0.2, Name "Cylinder center distance from bottom"}
jet_radius = {0.05, Name "Cylinder radius"}
jet_width = {10*Pi/180, Name "Jet width in radians"}
width = {0.41, Name "Channel width"}
cylinder_size = {0.02, Name "Mesh size on cylinder"}
box_size = {0.05, Name "Mesh size on wall"}
coarse_size = {0.1, Name "Mesh size close to the outflow"}
coarse_distance = {0.5, Name "Distance from the cylinder where coarsening starts"}
];

// Seed the cylinder
center = newp;
Point(center) = {0, 0, 0, cylinder_size};

n = #jet_positions[];

radius = jet_radius;

If(n > 0)
  cylinder[] = {};
  lower_bound[] = {};
  uppper_bound[] = {};

  //  Define jet surfaces
  For i In {0:(n-1)}

      angle = jet_positions[i];
  
      x = radius*Cos(angle-jet_width/2);
      y = radius*Sin(angle-jet_width/2);
      p = newp;
      Point(p) = {x, y, 0, cylinder_size};
      lower_bound[] += {p};

      x0 = radius*Cos(angle);
      y0 = radius*Sin(angle);
      arch_center = newp;
      Point(arch_center) = {x0, y0, 0, cylinder_size};

      x = radius*Cos(angle+jet_width/2);
      y = radius*Sin(angle+jet_width/2);
      q = newp;
      Point(q) = {x, y, 0, cylinder_size};
      upper_bound[] += {q};
  
      // Draw the piece; p to angle
      l = newl;
      Circle(l) = {p, center, arch_center}; 
      // Let each yet be marked as a different surface
      Physical Line(5+i) = {l};
      cylinder[] += {l};

      // Draw the piece; angle to q
      l = newl;
      Circle(l) = {arch_center, center, q}; 
      // Let each yet be marked as a different surface
      Physical Line(5+i) += {l};
      cylinder[] += {l};
  EndFor

  // Fill in the rest of the cylinder. These are no slip surfaces
  lower_bound[] += {lower_bound[0]};
  Physical Line(4) = {};  // No slip cylinder surfaces
  For i In {0:(n-1)}
    p = upper_bound[i];
    q = lower_bound[i+1];

    pc[] = Point{p}; // Get coordinates
    qc[] = Point{q}; // Get coordinates

    // Compute the angle
    angle_p = Atan2(pc[1], pc[0]);
    angle_p = (angle_p > 0) ? angle_p : (2*Pi + angle_p);

    angle_q = Atan2(qc[1], qc[0]);
    angle_q = (angle_q > 0) ? angle_q : (2*Pi + angle_q);

    angle = angle_q - angle_p; // front back
    angle = (angle < 0) ? angle + 2*Pi : angle; // check also back front
    Printf("%g", angle);
    // Greter than Pi, then we need to insert point
    If(angle > Pi)
      half[] = Rotate {{0, 0, 1}, {0, 0, 0}, angle/2} {Duplicata{Point{p};}};         

      l = newl;
      Circle(l) = {p, center, half}; 
      // Let each yet be marked as a different surface
      Physical Line(4) += {l};
      cylinder[] += {l};

      l = newl;
      Circle(l) = {half, center, q}; 
      // Let each yet be marked as a different surface
      Physical Line(4) += {l};
      cylinder[] += {l};                     
    Else
      l = newl;
      Circle(l) = {p, center, q}; 
      // Let each yet be marked as a different surface
      Physical Line(4) += {l};
      cylinder[] += {l};
    EndIf
  EndFor
// Just the circle
Else
   p = newp; 
   Point(p) = {-jet_radius, 0, 0, cylinder_size};
   Point(p+1) = {0, jet_radius, 0, cylinder_size};
   Point(p+2) = {jet_radius, 0, 0, cylinder_size};
   Point(p+3) = {0, -jet_radius, 0, cylinder_size};
	
   l = newl;
   Circle(l) = {p, center, p+1};
   Circle(l+1) = {p+1, center, p+2};
   Circle(l+2) = {p+2, center, p+3};
   Circle(l+3) = {p+3, center, p};

   cylinder[] = {l, l+1, l+2, l+3};			
   Physical Line(4) = {cylinder[]};
EndIf

// The chanel
p = newp;

Point(p) = {-front_distance, -bottom_distance, 0, box_size};
Point(p+1) = {jet_radius+coarse_distance, -bottom_distance, 0, coarse_size};
Point(p+2) = {-front_distance+length, -bottom_distance, 0, coarse_size};
Point(p+5) = {-front_distance, -bottom_distance+width, 0, box_size};
Point(p+4) = {jet_radius+coarse_distance, -bottom_distance+width, 0, coarse_size};
Point(p+3) = {-front_distance+length, -bottom_distance+width, 0, coarse_size};

l = newl;
// A no slip wall
Line(l) = {p, p+1};
Line(l+1) = {p+1, p+2};
Physical Line(1) = {l, l+1};

// Outflow
Line(l+2) = {p+2, p+3};
Physical Line(2) = {l+2};

// Top no slip wall
Line(l+3) = {p+3, p+4};
Line(l+4) = {p+4, p+5};
Physical Line(1) += {l+3, l+4};

// Inlet
Line(l+5) = {p+5, p};
Physical Line(3) = {l+5};

// Coarse line
Line(l+6) = {p+1, p+4};

coarse = newll;
Line Loop(coarse) = {(l+1), (l+2), (l+3), -(l+6)};

s = news;
Plane Surface(s) = {coarse};
Physical Surface(1) = {s};

// The one with cylinder 
cframe[] = {l, (l+6), l+4, l+5};

// // The surface to be mesh;
outer = newll;
Line Loop(outer) = {cframe[]};

inner = newll;
Line Loop(inner) = {cylinder[]};

s = news;
Plane Surface(s) = {inner, outer};
Physical Surface(1) += {s};

//Characteristic Length{cylinder[]} = cylinder_size;
//Characteristic Length{coarse[]} = coarse_size;
//Characteristic Length{cframe[]} = box_size;