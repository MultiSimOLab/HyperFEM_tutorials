#***********************************************************************************************
# Numerical simulation of a fully coupled magneto-mechanical model for Dowsil-based hMREs under a 
# simplified magnetic-field assumption
# Carlos Perez-Garcia, Rogelio Ortigosa, Jesus Martinez-Frutos Daniel Garcia-Gonzalez.
# Topology and material optimization in ultra-soft magnetoactive structures: making advantage of 
# residual anisotropies
# Advanced Materials
#***********************************************************************************************

# Load required Packages
using HyperFEM
using Gridap, GridapGmsh, GridapSolvers, DrWatson
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces


function invU(F; ε=1e-12)
  F_ = get_array(F)
  I2_ = get_array(I2)
  # Compute symmetric SPD = F'F and regularize
  S = F_' * F_
  Sreg = S + ε * I2_         # ensure SPD
  Sroot = sqrt(Sreg)          # matrix square root    
  # Guard against reflection (ensure det(R) ~ +1)
  return TensorValue(inv(Sroot))
end
 

# Initialize problem
pname = "hMRE_2D"
meshfile = "Short_magnet.msh"
simdir = datadir("sims", pname)
setupfolder(simdir)

# Load mesh file
geomodel = GmshDiscreteModel(datadir("models", meshfile))
#********************
# Constitutive models
#********************

# Mechanical model for Solid hMREs movement
params = [10.0, 12480.64495232286, 1.0, 2.0, 5195.545287237134, 0.2602717127043121, 2.0]
modelmech = NonlinearMooneyRivlin2D(λ=(params[1] + params[2]) * 1e2, μ1=params[1] * 1e3, μ2=params[2], α1=params[3], α2=params[4]) +
            TransverseIsotropy2D(μ=params[5], α1=params[6], α2=params[7])

# Magnetic parameters
μ0 = 4π * 1e-7
Banorm = 80.0E-9
αr = 10e-3 / μ0
#**********************
# Finite Elements
#**********************
order_solid = 1       # Linear FEs for solid
reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_solid)

#**********************
# Domains
#**********************
Ωsolid = Triangulation(geomodel, tags=["Beam"])
dΩsolid = Measure(Ωsolid, 2 * order_solid)

#************************************
# Dirichlet boundary conditions
#************************************

dir_u_tags_solid = ["Fixed_Solid", "Fixed_Solid_x"]
dir_u_values_solid = [[0.0, 0.0], [0.0, 0.0]]
Du_solid = DirichletBC(dir_u_tags_solid, dir_u_values_solid)

#******************************************
# Finite Element Spaces and state variables
#******************************************

Vu_solid = TestFESpace(Ωsolid, reffeu, Du_solid, conformity=:H1, dirichlet_masks=[(true, true), (true, false)])
Uu_solid⁺ = TrialFESpace(Vu_solid, Du_solid)
Uu_solid⁻ = TrialFESpace(Vu_solid, Du_solid)
uh_solid⁺ = FEFunction(Uu_solid⁺, zero_free_values(Uu_solid⁺))
uh_solid⁻ = FEFunction(Uu_solid⁻, zero_free_values(Uu_solid⁻))

# ******************************************************
#               Weak Forms
#******************************************************

# Derivatives of the energy functions
_, ∂Ψs∂F, ∂Ψs∂FF = modelmech()

# Kinematic functions
F, _, _= get_Kinematics(Kinematics(Mechano, Solid))

# Magnetization field
V_N = TestFESpace(Ωsolid, reffeu)
Nh = interpolate_everywhere((x) -> VectorValue(1.0, 0.0), V_N)
Bah = interpolate_everywhere(VectorValue([0.0, -Banorm]), V_N)
∂Ψmag∂F(Fn, N, Ba) = (-αr / μ0) * (Ba ⊗ ((invU(Fn)) * N))

res_mech(Λ) = (u, v) -> ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)'), Nh))))dΩsolid +
                        Λ * ∫((∂Ψmag∂F ∘ (F ∘ (∇(uh_solid⁻)'), Nh, Bah)) ⊙ (∇(v)'))dΩsolid

jac_mech(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'), Nh)) ⊙ (∇(du)')))dΩsolid 

# ******************************************************
#             Computational problems
#******************************************************

nls_mech = NewtonSolver(LUSolver(); maxiter=20, atol=1.e-8, rtol=1.e-2, verbose=true)
comp_model = StaticNonlinearModel(res_mech, jac_mech, Uu_solid⁺, Vu_solid, Du_solid; nls=nls_mech, xh=uh_solid⁺, xh⁻=uh_solid⁻)

#******************************************************
#               Run solver
#******************************************************

solve!(comp_model; stepping=(nsteps=12, maxbisec=1))

writevtk(Ωsolid, simdir * "/hMRE2", cellfields=["uh" => uh_solid⁺]) # MRE deformation