# Load required Packages
using HyperFEM
using Gridap, GridapGmsh, GridapSolvers, DrWatson
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using Gridap.CellData

# Initialize problem
pname    = "C_contact"
meshfile = "C_contact_surrounded.msh"
simdir    = datadir("sims", pname)
setupfolder(simdir)

# Load mesh file
geomodel = GmshDiscreteModel(datadir("models", meshfile))
#********************
# Constitutive models
#********************
# mechanical model of third medium (As Rigid As Possible)
fc             =  1e-8 # material parameter ratio bwtween third medium and solid
model_contact_ =  ARAP2D(μ=1e5*fc)
model_contact  =  HessianRegularization(mechano=model_contact_, δ=fc*1e5*1e-3)

# mechanical model of solid
modelmech   =  NonlinearMooneyRivlin2D(λ=2e7, μ1=1e5, μ2=1e5, α1=1.0, α2=2.0)

#**********************
# Finite Elements
#**********************
order_solid   =  1       # Linear FEs for solid
reffeu        =  ReferenceFE(lagrangian, VectorValue{2,Float64}, order_solid)

#**********************
# Domains
#**********************
Ωdomain     = Triangulation(geomodel, tags=["domain","third"])
Ωsolid      = Triangulation(geomodel, tags=["domain"])
Ωcontact    = Triangulation(geomodel, tags=["third"])
Γneumann    = BoundaryTriangulation(geomodel, tags="Neumann")

dΩdomain     = Measure(Ωdomain, 2 * order_solid)
dΩsolid      = Measure(Ωsolid, 2 * order_solid)
dΩcontact     = Measure(Ωcontact, 2 * order_solid)
dΓNeumann     = Measure(Γneumann, 2 * order_solid)

#************************************
# Dirichlet boundary conditions
#************************************
dir_u_tags = ["Fixed"]
dir_u_values = [[0.0, 0.0]]
Du = DirichletBC(dir_u_tags, dir_u_values)

#******************************************
# Finite Element Spaces and state variables
#******************************************
V   = TestFESpace(Ωdomain, reffeu, Du, conformity=:H1, dirichlet_masks=[(true, true)])
U   = TrialFESpace(V, Du)

# ******************************************************
#               Weak Forms
#******************************************************
# Derivatives of the energy functions
Ψc, ∂Ψc∂F,   ∂Ψc∂FF = model_contact()  
Ψs, ∂Ψs∂F,   ∂Ψs∂FF = modelmech()          

# Kinematic functions
F, H, J = get_Kinematics(Kinematics(Mechano, Solid))

t =  VectorValue(0.0,-6.0e4)
res(Λ) = (u, v) -> ∫((∇(v)' ⊙ (∂Ψs∂F ∘ (F ∘ (∇(u)') ))))dΩsolid +
                        ∫((∇(v)' ⊙ (∂Ψc∂F ∘ (F ∘ (∇(u)')))))dΩcontact - Λ*∫(t⋅v)dΓNeumann

jac(Λ) = (u, du, v) -> ∫(∇(v)' ⊙ ((∂Ψs∂FF ∘ (F ∘ (∇(u)'))) ⊙ (∇(du)')))dΩsolid+
                            ∫(∇(v)' ⊙ ((∂Ψc∂FF ∘ (F ∘ (∇(u)'))) ⊙ (∇(du)')))dΩcontact

# ******************************************************
#             Computational problem
#******************************************************

α = CellState(1.0, dΩdomain)
linesearch = Injectivity_Preserving_LS(α, U, V; maxiter=50, αmin=1e-16, ρ=0.5, c=0.95)
nls_mech = Newton_RaphsonSolver(LUSolver(); maxiter=500, rtol=1e-3, verbose=true, linesearch=linesearch)
comp_model = StaticNonlinearModel(res, jac, U, V, Du; nls=nls_mech)

# ******************************************************
#            Solver and Postprocessing
#******************************************************

function driverpost_mech(post;)
    Λ_ = post.iter
    Λ  = post.Λ[Λ_]
    state = post.comp_model.caches[3]
    uh = FEFunction(U, state)
    pvd = post.cachevtk[3]
    filePath = post.cachevtk[2]
    if Λ_ % 50 == 0
        Λstring = replace(string(round(Λ, digits=2)), "." => "_")
        pvd[Λ_] = createvtk(Ωsolid,
            filePath * "/contact_Λ_" * Λstring * "_TIME_$Λ_" * ".vtu",
            cellfields=["uh"=>uh])
    end
end
 
post_model     = PostProcessor(comp_model, driverpost_mech; is_vtk=true, filepath=simdir)

solve!(comp_model; stepping=(nsteps=500, maxbisec=10), :post => post_model)