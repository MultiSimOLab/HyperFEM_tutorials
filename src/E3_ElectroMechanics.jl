using HyperFEM
using Gridap, GridapGmsh, GridapSolvers, DrWatson
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using WriteVTK


simdir = datadir("sims", "Static_ElectroMechanical")
setupfolder(simdir)

geomodel = GmshDiscreteModel("./data/models/test_static_EM.msh")

# Constitutive model
physmodel_mec = NeoHookean3D(λ=10.0, μ=1.0)
physmodel_elec = IdealDielectric(ε=1.0)
physmodel= ElectroMechModel(mechano=physmodel_mec, electro=physmodel_elec)

# Functionals for Energy and Analytical derivatives
Ψ,∂ΨF, ∂ΨE, ∂ΨFF,∂ΨEF,∂ΨEE   = physmodel()

# Setup integration
order = 1
degree = 2 * order
Ω = Triangulation(geomodel)
dΩ = Measure(Ω, degree)

# Dirichlet boundary conditions 
evolu(Λ) = 1.0
dir_u_tags = ["fixedup"]
dir_u_values = [[0.0, 0.0, 0.0]]
dir_u_timesteps = [evolu]
Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

evolφ(Λ) = Λ
dir_φ_tags = ["midsuf", "topsuf"]
dir_φ_values = [0.0, 0.1]
dir_φ_timesteps = [evolφ, evolφ]
Dφ = DirichletBC(dir_φ_tags, dir_φ_values, dir_φ_timesteps)

D_bc = MultiFieldBC([Du, Dφ])

# Finite Elements
reffeu = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
reffeφ = ReferenceFE(lagrangian, Float64, order)

# Test FE Spaces
Vu = TestFESpace(geomodel, reffeu, D_bc.BoundaryCondition[1], conformity=:H1)
Vφ = TestFESpace(geomodel, reffeφ, D_bc.BoundaryCondition[2], conformity=:H1)

# Trial FE Spaces
Uu = TrialFESpace(Vu, D_bc.BoundaryCondition[1], 1.0)
Uφ = TrialFESpace(Vφ, D_bc.BoundaryCondition[2], 1.0)

# Multifield FE Spaces
V = MultiFieldFESpace([Vu, Vφ])
U = MultiFieldFESpace([Uu, Uφ])

# Kinematic Description
km=Kinematics(Mechano,Solid)
ke=Kinematics(Electro,Solid)

F,_,_ = get_Kinematics(km)
E     = get_Kinematics(ke)

# residual and jacobian function of load factor
res(Λ)= ((u, φ), (v, vφ)) ->   ∫(∇(v)' ⊙ (∂ΨF ∘ (F∘∇(u)', E∘∇(φ))))dΩ -
                                ∫(∇(vφ) ⋅ (∂ΨE ∘ (F∘∇(u)', E∘∇(φ))))dΩ

jac(Λ)= ((u, φ), (du, dφ), (v, vφ)) -> ∫(∇(v)' ⊙ ((∂ΨFF ∘ (F∘∇(u)', E∘∇(φ))) ⊙ ∇(du)'))dΩ +
                                        ∫(∇(vφ)' ⋅ ((∂ΨEE ∘ (F∘∇(u)', E∘∇(φ))) ⋅ ∇(dφ)))dΩ -
                                        ∫(∇(dφ) ⋅ ((∂ΨEF ∘ (F∘∇(u)', E∘∇(φ))) ⊙ ∇(v)'))dΩ -
                                        ∫(∇(vφ) ⋅ ((∂ΨEF ∘ (F∘∇(u)', E∘∇(φ))) ⊙ ∇(du)'))dΩ 
# nonlinear solver
ls = LUSolver()
nls_ = NewtonSolver(ls; maxiter=20, atol=1.e-10, rtol=1.e-8, verbose=true)

# Computational model
comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls=nls_)

# Postprocessor to save results
function driverpost(post; Ω=Ω, U=U)
    state = post.comp_model.caches[3]
    Λ_ = post.iter

    xh = FEFunction(U, state)
    uh = xh[1]
    φh = xh[2]
    pvd = post.cachevtk[3]
    filePath = post.cachevtk[2]

    if post.cachevtk[1]
        pvd[Λ_] = createvtk(Ω,filePath * "/TIME_$Λ_" * ".vtu",cellfields=["u" => uh, "φ" => φh])
    end
end

post_model = PostProcessor(comp_model, driverpost; is_vtk=true, filepath=simdir)

# Solve
x = solve!(comp_model; stepping=(nsteps=5, maxbisec=5), ProjectDirichlet=true, post=post_model)