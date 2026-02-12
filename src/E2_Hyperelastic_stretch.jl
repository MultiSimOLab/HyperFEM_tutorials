using HyperFEM, Gridap, TimerOutputs, DrWatson

# Initialize problem
pname    = "Hyperelastic_stretch"
simdir    = datadir("sims", pname)
setupfolder(simdir)

# Model
domain = (0,1,0,0.2,0,0.5)
partition = (20,2,10)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"diri_0",[25, 19, 17, 15, 13, 1, 3 ,5, 7])
add_tag_from_tags!(labels,"diri_1",[26, 20 ,18 ,16 ,14 ,2 ,4 ,6, 8])

modelmech   =  NeoHookean3D(λ=100.0, μ=1.0)
  
#**********************
# Finite Elements
#**********************
order_solid   =  1       # Linear FEs for solid
reffeu        =  ReferenceFE(lagrangian, VectorValue{3,Float64}, order_solid)

#**********************
# Domains
#**********************
Ω    = Triangulation(model)
dΩ = Measure(Ω,2*order_solid)

#************************************
# Dirichlet boundary conditions
#************************************
dir_u_tags = ["diri_0", "diri_1"]
dir_u_values = [[0.0, 0.0, 0.0],[0.75, 0.0, 0.0]]
dir_u_timesteps = [(Λ)->1.0, (Λ)->Λ]
Du = DirichletBC(dir_u_tags, dir_u_values,dir_u_timesteps)

#******************************************
# Finite Element Spaces and state variables
#******************************************
V   = TestFESpace(Ω, reffeu, Du, conformity=:H1)
U   = TrialFESpace(V, Du)

# ******************************************************
#               Weak Forms
#******************************************************
# Derivatives of the energy functions
Ψ, ∂Ψ∂F, ∂Ψ∂FF = modelmech()  

# Kinematic functions
F, H, J = get_Kinematics(Kinematics(Mechano, Solid))

res(Λ) = (u, v) -> ∫((∇(v)' ⊙ (∂Ψ∂F ∘ (F ∘ (∇(u)') ))))dΩ
jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψ∂FF ∘ (F ∘ (∇(u)'))) ⊙ (∇(du)')))dΩ

# ******************************************************
#             Computational problem
#******************************************************
nls_mech = Newton_RaphsonSolver(LUSolver(); maxiter=20, rtol=1e-4, verbose=true)
compmodel = StaticNonlinearModel(res, jac, U, V, Du; nls=nls_mech)
uh=get_state(compmodel) 

# ******************************************************
#            Solver and Postprocessing
#******************************************************

@time solve!(compmodel; stepping=(nsteps=5, maxbisec=1), ProjectDirichlet=true)

writevtk(Ω, simdir*"/hyperelastic", cellfields = ["uh" => uh])