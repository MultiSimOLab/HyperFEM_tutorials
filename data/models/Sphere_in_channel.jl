using Gmsh: Gmsh, gmsh
Gmsh.initialize()
gmsh.model.add("Sphere_in_channel")

geo = gmsh.model.geo

# parameters
L   =  20
H   =  9
R   =  0.5

lc  =  L/100


#------------------------------
# Generation of points in the rectangle
#------------------------------
geo.addPoint(-L/2, -H/2, 0,  lc, 1) 
geo.addPoint(L/2, -H/2, 0,  lc, 2) 
geo.addPoint(L/2, H/2, 0,  lc, 3) 
geo.addPoint(-L/2, H/2, 0,  lc, 4) 

geo.addLine(1,2,1)
geo.addLine(2,3,2)
geo.addLine(3,4,3)
geo.addLine(4,1,4)

geo.addCurveLoop([1,2,3,4], 1)

#------------------------------
# Circle
#------------------------------
geo.addPoint(-L/4, 0.0, 0,  lc/5, 5) 
geo.addPoint(R-L/4, 0.0, 0,  lc/5, 6) 
geo.addPoint(-L/4, R, 0,  lc/5, 7) 
geo.addPoint(-R-L/4, 0.0, 0,  lc/5, 8) 
geo.addPoint(-L/4, -R, 0,  lc/5, 9) 

geo.addCircleArc(6,5,7)
geo.addCircleArc(7,5,8)
geo.addCircleArc(8,5,9)
geo.addCircleArc(9,5,6)

geo.addCurveLoop([5,6,7,8], 2)

geo.addPlaneSurface([1,2], 1)

#-----------------------------------
#-----------------------------------
#-----------------------------------
# Meshing and physical tags
#-----------------------------------
#-----------------------------------
#----------------------------------- 
gmsh.model.addPhysicalGroup(0, [2,3], 1,"Dirichlet_Wall")  
gmsh.model.addPhysicalGroup(0, [1,4], 2,"Dirichlet_Inlet")  
gmsh.model.addPhysicalGroup(0, [6,7,8,9], 3,"Dirichlet_Circle")  

gmsh.model.addPhysicalGroup(1, [5,6,7,8], 1,"Dirichlet_Circle")  
gmsh.model.addPhysicalGroup(1, [1,3], 2,"Dirichlet_Wall")  
gmsh.model.addPhysicalGroup(1, [4], 3,"Dirichlet_Inlet")  

gmsh.model.addPhysicalGroup(2, [1], 1,"Domain")  

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate() 


#-----------------------------------
#-----------------------------------
#-----------------------------------
# Saving
#-----------------------------------
#-----------------------------------
#-----------------------------------
gmsh.model.geo.synchronize()
output_file = joinpath(dirname(@__FILE__), "Sphere_in_channel.msh")
gmsh.write(output_file)
 # Launch the GUI to see the results:
if !("-nopopup" in ARGS)
        gmsh.fltk.run()
end
Gmsh.finalize()

#end