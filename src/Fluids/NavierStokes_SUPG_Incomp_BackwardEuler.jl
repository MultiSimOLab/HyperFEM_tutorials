using HyperFEM
using Gridap, GridapGmsh, GridapSolvers, DrWatson
using GridapSolvers.NonlinearSolvers
using Gridap.FESpaces
using WriteVTK
using TimerOutputs


function main(nsteps, tend)

    simdir = datadir("sims", "fluid_NavierStokes_SUPG_Inc")
    setupfolder(simdir)

    # Physical properties
    μ = 0.01
    ρ = 1

    geomodel = GmshDiscreteModel("./data/models/Sphere_in_channel.msh")
    # Setup integration
    orderu = 1
    orderp = 1

    Ω = Triangulation(geomodel)
    dΩ = Measure(Ω, 4 * orderu)

    # Finite Elements
    reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, orderu)
    reffep = ReferenceFE(lagrangian, Float64, orderp)
    reffel2 = ReferenceFE(lagrangian, Float64, 0)
    reffegu = ReferenceFE(lagrangian, TensorValue{2,2,Float64}, orderu - 1)

    # DirichletBC velocity
    CFL = 0.3
    Reynolds = 100
    D = 1.0
    U0 = Reynolds * μ / (D * ρ)
    cellmeasure = sqrt.(get_cell_measure(Ω))
    Vl2 = TestFESpace(Ω, reffel2, conformity=:L2)
    he = FEFunction(Vl2, cellmeasure)
    # const Δt = CFL*minimum(he.free_values)/U0
    @show Reynolds = D * U0 * ρ / μ
    Δt = tend / nsteps

    evolu(t) = max(1, (3 / tend) * t)
    dir_u_tags = ["Dirichlet_Inlet", "Dirichlet_Wall", "Dirichlet_Circle"]
    dir_u_values = [x -> VectorValue(U0, 0.0), [0.0, 0.0], [0.0, 0.0]]
    dir_u_timesteps = [evolu, evolu, evolu]
    Du = DirichletBC(dir_u_tags, dir_u_values, dir_u_timesteps)

    D_bc = MultiFieldBC([Du, NothingBC()])

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

    ε(∇u) = 0.5 * (∇u + ∇u')
    a(u, un) = (u - un) / Δt
    Da(Δu) = Δu / Δt
    τlsic(u, he) = ρ * norm(u) * he * 0.5
    τpspg(he) = (1 / 12) * he^2 * (1 / μ)
    # τsupg(u, he) = ((2 * norm(u)^2 / he) + 9 * (4 * μ / (ρ * he^2))^2 + (1 / Δt)^2)^-1 / 2
    τsupg(u, he) = ((2 * norm(u) / he)^2 + 9 * (4 * μ / (ρ * he^2))^2 + (2 / Δt)^2)^-1 / 2

    Ru(u, a, ∇p, ∇u, ∇2u⁻) = ρ * a - μ * ∇2u⁻ + ρ * ∇u * u + ∇p
    DRu_u(u, ∇u, Δu, ∇Δu) = ρ * Da(Δu) + ρ * ∇Δu * u + ρ * ∇u * Δu
    DRu_p(∇Δp) = ∇Δp

    ∇u⁻ = interpolate_everywhere(∇(xh⁻[1])', Vgu)

    resu_gk((u, p), (v, q)) = ∫(ρ * v ⋅ (a ∘ (u, xh⁻[1])))dΩ +
                              ∫((μ * (ε ∘ (∇(u)))) ⊙ (ε ∘ (∇(v))))dΩ -
                              ∫(p * (∇ ⋅ v))dΩ +
                              ∫(ρ * (∇(u)' * u) ⋅ v)dΩ
    resu_τ((u, p), (v, q)) = ∫((τsupg ∘ (xh⁻[1], he)) * (∇(v)' * u) ⋅ (Ru ∘ (u, (a ∘ (u, xh⁻[1])), ∇(p), ∇(u)', (∇ ⋅ (∇u⁻)))))dΩ +
                             ∫((τlsic ∘ (xh⁻[1], he)) * (∇ ⋅ v) * (∇ ⋅ u))dΩ
    resp_gk((u, p), (v, q)) = -1.0 * ∫(q * (∇ ⋅ u))dΩ
    resp_τ((u, p), (v, q)) = -1.0 * ∫((τpspg ∘ (he)) * ∇(q) ⋅ (Ru ∘ (u, (a ∘ (u, xh⁻[1])), ∇(p), ∇(u)', (∇ ⋅ (∇u⁻)))))dΩ

    res(Λ) = ((u, p), (v, q)) -> resu_gk((u, p), (v, q)) +
                                 resu_τ((u, p), (v, q)) +
                                 resp_gk((u, p), (v, q)) +
                                 resp_τ((u, p), (v, q))


    jacu_gk((u, p), (du, dp), (v, q)) = ∫(ρ * v ⋅ (Da ∘ (du)))dΩ +
                                        ∫((μ * (ε ∘ (∇(du)))) ⊙ (ε ∘ (∇(v))))dΩ -
                                        ∫(dp * (∇ ⋅ v))dΩ +
                                        ∫(ρ * (∇(du)' * u) ⋅ v)dΩ +
                                        ∫(ρ * (∇(u)' * du) ⋅ v)dΩ

    jacp_gk((u, p), (du, dp), (v, q)) = -1.0 * ∫(q * (∇ ⋅ du))dΩ

    jacu_τ((u, p), (du, dp), (v, q)) = ∫((τsupg ∘ (xh⁻[1], he)) * ((∇(v)' * du) ⋅ (Ru ∘ (u, (a ∘ (u, xh⁻[1])), ∇(p), ∇(u)', (∇ ⋅ (∇u⁻)))) +
                                                                   (∇(v)' * u) ⋅ (DRu_u ∘ (u, ∇(u)', du, ∇(du)')) +
                                                                   (∇(v)' * u) ⋅ (DRu_p ∘ (∇(dp)))))dΩ +
                                       ∫((τlsic ∘ (xh⁻[1], he)) * (∇ ⋅ v) * (∇ ⋅ du))dΩ


    jacp_τ((u, p), (du, dp), (v, q)) = -1.0 * ∫((τpspg ∘ (he)) * ∇(q) ⋅ (DRu_u ∘ (u, ∇(u)', du, ∇(du)')))dΩ -
                                       ∫((τpspg ∘ (he)) * ∇(q) ⋅ (DRu_p ∘ (∇(dp))))dΩ


    jac(Λ) = ((u, p), (du, dp), (v, q)) -> jacu_gk((u, p), (du, dp), (v, q)) +
                                           jacu_τ((u, p), (du, dp), (v, q)) +
                                           jacp_gk((u, p), (du, dp), (v, q)) +
                                           jacp_τ((u, p), (du, dp), (v, q))


    ls = LUSolver()
    nls_ = NewtonSolver(ls; maxiter=20, atol=1.e-10, rtol=1.e-8, verbose=true)

    # Computational model
    compmodel = StaticNonlinearModel(res, jac, U⁺, V, D_bc; nls=nls_, xh⁻=xh⁻)

    xh⁺ = get_state(compmodel)
    function driverpost(post)
        @show post.iter
        Λ_ = post.iter
        Λ = post.Λ[Λ_]
        uh = xh⁺[1]
        ph = xh⁺[2]
        pvd = post.cachevtk[3]
        filePath = post.cachevtk[2]
        if post.cachevtk[1] && mod(Λ_, 50) == 0
            pvd[Λ_] = createvtk(Ω, filePath * "/$(Λ)_Stokes" * ".vtu", cellfields=["u" => uh, "p" => ph])
        end
    end

    post_model = PostProcessor(compmodel, driverpost; is_vtk=true, filepath=simdir)

    for (i, t) in enumerate(range(0.0, stop=tend, length=nsteps + 1))
        @show t
        TrialFESpace!(U⁺, compmodel.dirichlet, t)
        x = solve!(compmodel; stepping=(nsteps=1, maxbisec=0))
        # @show Reynolds = 0.15 *maximum(xh⁺[1])* ρ/μ
        TrialFESpace!(U⁻, compmodel.dirichlet, t)
        get_free_dof_values(xh⁻) .= get_free_dof_values(xh⁺)
        interpolate_everywhere!(∇(xh⁻[1])', get_free_dof_values(∇u⁻), ∇u⁻.dirichlet_values, Vgu)
        post_model(t)
    end
    HyperFEM.vtk_save(post_model)
end

reset_timer!()
main(4800, 144)
print_timer()