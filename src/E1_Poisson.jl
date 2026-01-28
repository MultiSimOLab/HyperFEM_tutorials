using HyperFEM, Gridap, WriteVTK, DrWatson

simdir = datadir("sims", "poisson")
setupfolder(simdir)

########################################
# EXACT SOLUTION AND DIFFERENTIAL OPERATORS
########################################

# Exact solution, Analytical gradient, Analytical Laplacian  of the Poisson problem
u₀(x) = cos(x[1]) * sin(x[2] + π)
∇u₀(x) = VectorValue(-sin(x[1]) * sin(x[2] + π),cos(x[1]) * cos(x[2] + π),0.0)
Δu₀(x) = -2.0 * cos(x[1]) * sin(x[2] + π)

########################################
# PROBLEM DEFINITION
########################################

# Source term of the Poisson equation:
#   -Δu = f
# Defined so that u₀ is the exact solution
f(x) = -Δu₀(x)

# Homogeneous Neumann boundary condition
# On the corresponding faces, the outward normal is ±e₃
# Since ∂u₀/∂x₃ = 0, the normal flux vanishes
h(x) = 0.0

########################################
# DOMAIN AND MESH
########################################

# Three-dimensional computational domain:
# x ∈ (-π, π), y ∈ (-π/2, π/2), z ∈ (0, 1)
domain = (-π, π, -π/2, π/2, 0, 1)

# Number of cells in each spatial direction
# Anisotropic mesh consistent with the solution dependence
nC = (100,40, 5)

# Cartesian discrete model
model = CartesianDiscreteModel(domain, nC)

########################################
# BOUNDARY LABELING
########################################

# Retrieve the automatic face labeling of the model
labels = get_face_labeling(model)

# Assign Neumann boundary faces
add_tag_from_tags!(labels, "neumann", ["tag_21", "tag_22"])

# Assign Dirichlet boundary faces
add_tag_from_tags!(labels, "dirichlet", ["tag_23", "tag_24", "tag_25", "tag_26"])

# DirichletBC
dir_u_tags = ["dirichlet"]
dir_u_values = [x->u₀(x)]
Du = DirichletBC(dir_u_tags, dir_u_values)

########################################
# FINITE ELEMENT SPACES
########################################

# Polynomial order of the finite element approximation
order = 1
# Lagrangian finite element of order 'order'
reffe = ReferenceFE(lagrangian, Float64, order)
# H¹-conforming test space with Dirichlet boundary conditions
V = TestFESpace(model, reffe, Du, conformity=:H1)
# Trial space with strong imposition of Dirichlet data
U = TrialFESpace(V, Du)

########################################
# NUMERICAL INTEGRATION
########################################

# Quadrature degree sufficient for polynomial products
degree = order * 2
# Domain triangulation and volume measure
Ω  = Triangulation(model)
dΩ = Measure(Ω, degree)

# Neumann boundary triangulation and boundary measure
Γ  = BoundaryTriangulation(model, tags = "neumann")
dΓ = Measure(Γ, degree)

########################################
# VARIATIONAL FORMULATION
########################################

# Bilinear form of the Poisson operator
# a(u,v) = ∫_Ω ∇u · ∇v
a(u, v) = ∫( ∇(v) ⋅ ∇(u) ) * dΩ

# Linear form:
# - volumetric source term
# - Neumann boundary contribution
l(v) = ∫( v * f ) * dΩ + ∫( v * h ) * dΓ

compmodel = StaticLinearModel(l, a, U, V, Du)
solve!(compmodel; Assembly=false)

uh=get_state(compmodel) 
 
########################################
# L2 ERROR COMPUTATION
########################################

# Higher quadrature degree for accurate error evaluation
dΩe = Measure(Ω, degree * 2)
# Pointwise error
e = uh - u₀
# L2 norm of the error
l2_error = sqrt(sum(∫( e ⋅ e ) * dΩe))
 
writevtk(Ω, simdir*"/poisson", cellfields = ["uh" => uh, "error"=> e])


# xh_= get_free_dof_values(h_)
# for  n in [2.0, 3.0, 4.0]
#    xh_ .+= n
#    solve!(compmodel)
#    e = uh - u₀
#    @show l2_error = sqrt(sum(∫( e ⋅ e ) * dΩe))
# end