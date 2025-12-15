using HyperFEM
using Gridap, GridapGmsh, GridapSolvers, DrWatson
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using WriteVTK
using TimerOutputs


function main(nsteps,Δt)
simdir = datadir("sims", "fluid_stokes_SUPG_backeuler")
setupfolder(simdir)

geomodel = GmshDiscreteModel("./data/models/Sphere_in_channel.msh")

# Setup integration
orderu = 2
orderp = 1

Ω = Triangulation(geomodel)
dΩ = Measure(Ω, 2 * orderu)

# DirichletBC velocity
evolu(Λ) = sin(Λ)
dir_u_tags = ["Dirichlet_Inlet", "Dirichlet_Wall", "Dirichlet_Circle"]
dir_u_values = [x -> VectorValue([16 * 1.0 * (0.0625 - x[2]^2), 0.0]), [0.0, 0.0], [0.0, 0.0]]
dir_u_timesteps = [evolu, evolu, evolu]
Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

D_bc = MultiFieldBC([Du, NothingBC()])

# Finite Elements
reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, orderu)
reffep = ReferenceFE(lagrangian, Float64, orderp)
reffel2 = ReferenceFE(lagrangian, Float64, 0)
reffegu = ReferenceFE(lagrangian, TensorValue{2,2,Float64}, orderu - 1)

# Test FE Spaces
Vu = TestFESpace(geomodel, reffeu, D_bc[1], conformity=:H1)
Vp = TestFESpace(geomodel, reffep, D_bc[2], conformity=:H1)
Vgu = TestFESpace(geomodel, reffegu, conformity=:L2)

# Trial FE Spaces
Uu⁺ = TrialFESpace(Vu, D_bc[1], 0.0)
Up⁺ = TrialFESpace(Vp, D_bc[2], 0.0)
Uu⁻ = TrialFESpace(Vu, D_bc[1], 0.0)
Up⁻ = TrialFESpace(Vp, D_bc[2], 0.0)

# Multifield FE Spaces
V = MultiFieldFESpace([Vu, Vp])
U⁺ = MultiFieldFESpace([Uu⁺, Up⁺])
U⁻ = MultiFieldFESpace([Uu⁻, Up⁻])

xh⁻ = FEFunction(U⁻, zero_free_values(U⁻))

Vl2 = TestFESpace(Ω, reffel2, conformity=:L2)
I = TensorValue(1.0, 0.0, 0.0, 1.0)
ε(∇u) = 0.5 * (∇u + ∇u')
μ = 1.0
ρ = 1.0
cellmeasure = sqrt.(get_cell_measure(Ω))
he = FEFunction(Vl2, cellmeasure)
τlsic(u, he) = ρ * norm(u) * he * 0.5
τpspg(he) = (1 / 12) * he^2 * (1 / μ)


gu = interpolate_everywhere(∇(xh⁻[1])', Vgu)

agk((u, p), (v, q)) = ∫((ρ/Δt)*(u⋅v))dΩ+∫((μ * (ε ∘ (∇(u))) - p * I) ⊙ (ε ∘ (∇(v))))dΩ - ∫(q * (I ⊙ (ε ∘ (∇(u)))))dΩ
aτ((u, p), (v, q)) = ∫((τlsic ∘ (xh⁻[1], he)) * (∇ ⋅ v) * (∇ ⋅ u))dΩ - ∫((τpspg ∘ (he)) * (∇(q) ⋅ ∇(p)))dΩ
a((u, p), (v, q)) = agk((u, p), (v, q)) + aτ((u, p), (v, q))
l((v, q)) = ∫((ρ/Δt)*(xh⁻[1]⋅v))dΩ-∫((τpspg ∘ (he)) * μ * (∇(q) ⋅ (∇ ⋅ (gu))))dΩ

compmodel = StaticLinearModel(l, a, U⁺, V, D_bc)

xh⁺ = get_state(compmodel)
function driverpost(post)
    Λ_ = post.iter
    Λ = post.Λ[Λ_]
    uh = xh⁺[1]
    ph = xh⁺[2]
    pvd = post.cachevtk[3]
    filePath = post.cachevtk[2]
    if post.cachevtk[1]
        pvd[Λ_] = createvtk(Ω, filePath * "/$(Λ)_Stokes" * ".vtu", cellfields=["u" => uh, "p" => ph])
    end
end

post_model = PostProcessor(compmodel, driverpost; is_vtk=true, filepath=simdir)

for (i, t) in enumerate(range(0.0, stop=nsteps*Δt, length=nsteps+1))
    @show t
    TrialFESpace!(U⁺, compmodel.dirichlet, t)
    solve!(compmodel; Assembly=true)
    TrialFESpace!(U⁻, compmodel.dirichlet, t)
    get_free_dof_values(xh⁻) .= get_free_dof_values(xh⁺)
    interpolate_everywhere!(∇(xh⁻[1])',get_free_dof_values(gu) ,gu.dirichlet_values ,Vgu)
    # post_model(t)
end
HyperFEM.vtk_save(post_model)
end

reset_timer!()
main(500,0.025)
print_timer()