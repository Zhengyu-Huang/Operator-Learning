cl1 = 0.1;
//+
Point(1) = {0, -0, 0, cl1 };
//+
Point(2) = {1, -0, 0, cl1 };
//+
Point(3) = {1, 1, 0, cl1 };
//+
Point(4) = {0, 1, 0, cl1 };
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};

ox = 0.5;
oy = 0.5;
r = 0.3;
cl2 = 0.1;
Point(5) = {ox + r, oy, 0, cl2};
//+
Point(6) = {ox, oy + r, 0, cl2};
//+
Point(7) = {ox - r, oy, 0, cl2};
//+
Point(8) = {ox, oy - r, 0, cl2};
//+
Point(9) = {0.5, 0.5, 0, 1.0};
//+
Circle(5) = {5, 9, 6};
//+
Circle(6) = {6, 9, 7};
//+
Circle(7) = {7, 9, 8};
//+
Circle(8) = {8, 9, 5};
//+
SetFactory("Built-in");
//+
Curve Loop(1) = {7, 8, 5, 6};
//+
Plane Surface(1) = {1};
//+
SetFactory("Built-in");
//+
Curve Loop(2) = {2, 3, 4, 1};
//+
Plane Surface(2) = {-1, -2};


Recombine Surface {1};

Recombine Surface {2};

//+
Physical Surface("Circle", 9) = {1};
//+
Physical Surface("Square", 10) = {2};
//+
Physical Curve("NeumannTop", 11) = {3};
//+
Physical Curve("DirichletBottom", 12) = {1};
