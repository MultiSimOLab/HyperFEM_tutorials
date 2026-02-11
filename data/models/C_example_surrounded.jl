using Gmsh: Gmsh, gmsh
Gmsh.initialize()
gmsh.model.add("MagnetoContact")

#function ThreeLayerModel(L,W,T,nl,nw,nt)

# parameters
const L=2;      
const H=1;      
const t=L*0.1;
fv    =  0.9;
fh    =  0.9;
const Lv=L*fv;      
const Hv=L*fh;      
const ref  =  1

lc     =  L/20
lv     =  L/20

geo = gmsh.model.geo
#-----------------------------------
#  Creating Nodes
#-----------------------------------
geo.addPoint(0.0, -Hv, 0,  lv, 1)
geo.addPoint(L+Lv, -Hv, 0,  lv, 2)


geo.addPoint(0.0, 0.0, 0,  lc, 3)
geo.addPoint(L, 0.0, 0,  lc, 4)

geo.addPoint(t, t, 0,  lc, 5)
geo.addPoint(L, t, 0,  lc, 6)

geo.addPoint(t, H-t, 0,  lc, 7)
geo.addPoint(L, H-t, 0,  lc, 8)

geo.addPoint(0, H, 0,  lc, 9)
geo.addPoint(L, H, 0,  lc, 10)

geo.addPoint(0.0, H+Hv, 0,  lv, 11)
geo.addPoint(L+Lv, H+Hv, 0,  lv, 12)

#-----------------------------------
#  Creating lines
#-----------------------------------
#bottom solid horizonal lines
geo.addLine(1, 2, 1)
geo.addLine(3, 4, 2)
geo.addLine(5, 6, 3)
geo.addLine(7, 8, 4)
geo.addLine(9, 10, 5)
geo.addLine(11, 12, 6)

geo.addLine(1, 3, 7)
geo.addLine(3, 9, 8)
geo.addLine(5, 7, 9)
geo.addLine(4, 6, 10)
geo.addLine(8, 10, 11)
geo.addLine(9, 11, 12)

geo.addLine(2, 12, 13)

#-----------------------------------
#  Creating curve loops and surfaces
#-----------------------------------
geo.addCurveLoop([2,10,-3,9,4,11,-5,-8], 1)
geo.addPlaneSurface([1], 1)
geo.addCurveLoop([1,13,-6,-12,5,-11,-4,-9,3,-10,-2,-7], 2)
geo.addPlaneSurface([2], 2)
gmsh.model.geo.synchronize()
#-----------------------------------
#-----------------------------------
# Meshing and physical tags
#-----------------------------------
#----------------------------------- 
gmsh.model.addPhysicalGroup(0, [3,9], 1, "Fixed")  
gmsh.model.addPhysicalGroup(1, [8], 1, "Fixed")  
gmsh.model.addPhysicalGroup(1, [11], 2, "Neumann")  
gmsh.model.addPhysicalGroup(2, [1], 1, "domain")  
gmsh.model.addPhysicalGroup(2, [2], 2, "third")  
gmsh.model.mesh.generate() 
#-----------------------------------
#-----------------------------------
# Saving
#-----------------------------------
#-----------------------------------
gmsh.model.geo.synchronize()
output_file = joinpath(dirname(@__FILE__), "C_contact_surrounded.msh")
gmsh.write(output_file)
 # Launch the GUI to see the results:
if !("-nopopup" in ARGS)
        gmsh.fltk.run()
end
Gmsh.finalize()

#end