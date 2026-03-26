using Test
using Odin

@testset "Categorical Extension" begin

    @testset "EpiNet construction" begin
        # Empty construction
        net = EpiNet()
        @test nspecies(net) == 0
        @test ntransitions(net) == 0

        # Add species
        add_species!(net, :S, 990.0)
        add_species!(net, :I, 10.0)
        add_species!(net, :R, 0.0)
        @test nspecies(net) == 3
        @test species_names(net) == [:S, :I, :R]
        @test species_concentrations(net) ≈ [990.0, 10.0, 0.0]

        # Add transitions
        add_transition!(net, :inf, [:S, :I] => [:I, :I], :beta)
        add_transition!(net, :rec, [:I] => [:R], :gamma)
        @test ntransitions(net) == 2
        @test transition_names(net) == [:inf, :rec]

        # Batch construction
        sir = EpiNet(
            [:S => 990.0, :I => 10.0, :R => 0.0],
            [:inf => ([:S, :I] => [:I, :I], :beta),
             :rec => ([:I] => [:R], :gamma)]
        )
        @test nspecies(sir) == 3
        @test ntransitions(sir) == 2
    end

    @testset "Stoichiometry" begin
        sir = SIR()
        S = stoichiometry_matrix(sir)
        @test size(S) == (3, 2)
        # Infection: S-1, I+1, R=0
        @test S[:, 1] == [-1, 1, 0]
        # Recovery: S=0, I-1, R+1
        @test S[:, 2] == [0, -1, 1]

        M = input_matrix(sir)
        @test size(M) == (3, 2)
        # Infection inputs: S=1, I=1
        @test M[:, 1] == [1, 1, 0]
        # Recovery inputs: I=1
        @test M[:, 2] == [0, 1, 0]
    end

    @testset "Input/output species" begin
        sir = SIR()
        @test Set(input_species(sir, :inf)) == Set([:S, :I])
        @test output_species(sir, :inf) == [:I, :I]  # Both output arcs go to I
        @test input_species(sir, :rec) == [:I]
        @test output_species(sir, :rec) == [:R]
    end

    @testset "Pre-built models" begin
        # SIR
        sir = SIR()
        @test nspecies(sir) == 3
        @test ntransitions(sir) == 2

        # SEIR
        seir = SEIR()
        @test nspecies(seir) == 4
        @test ntransitions(seir) == 3
        @test Set(species_names(seir)) == Set([:S, :E, :I, :R])

        # SIS
        sis = SIS()
        @test nspecies(sis) == 2
        @test ntransitions(sis) == 2

        # SIRS
        sirs = SIRS()
        @test nspecies(sirs) == 3
        @test ntransitions(sirs) == 3

        # SEIRS
        seirs = SEIRS()
        @test nspecies(seirs) == 4
        @test ntransitions(seirs) == 4

        # SIR + vaccination
        sir_v = SIRVax()
        @test nspecies(sir_v) == 4
        @test ntransitions(sir_v) == 3
    end

    @testset "Composition" begin
        infection = EpiNet([:S => 990.0, :I => 10.0],
                           [:inf => ([:S, :I] => [:I, :I], :beta)])
        recovery = EpiNet([:I => 10.0, :R => 0.0],
                          [:rec => ([:I] => [:R], :gamma)])
        sir = compose(infection, recovery)

        @test nspecies(sir) == 3
        @test ntransitions(sir) == 2
        @test species_names(sir) == [:S, :I, :R]
        @test transition_names(sir) == [:inf, :rec]

        # Stoichiometry should match direct SIR
        S_composed = stoichiometry_matrix(sir)
        S_direct = stoichiometry_matrix(SIR())
        @test S_composed == S_direct

        # Compose three parts: infection + progression + recovery (SEIR)
        inf_part = EpiNet([:S => 990.0, :E => 0.0, :I => 10.0],
                          [:inf => ([:S, :I] => [:E, :I], :beta)])
        prog_part = EpiNet([:E => 0.0, :I => 10.0],
                           [:prog => ([:E] => [:I], :sigma)])
        rec_part = EpiNet([:I => 10.0, :R => 0.0],
                          [:rec => ([:I] => [:R], :gamma)])
        seir = compose(inf_part, prog_part, rec_part)
        @test nspecies(seir) == 4
        @test ntransitions(seir) == 3
        @test Set(species_names(seir)) == Set([:S, :E, :I, :R])
    end

    @testset "Stratification" begin
        sir = SIR()

        # Basic stratification (no contact matrix)
        sir_2g = stratify(sir, [:young, :old])
        @test nspecies(sir_2g) == 6
        sn = species_names(sir_2g)
        @test :S_young in sn
        @test :I_old in sn
        @test :R_young in sn

        # With contact matrix
        C = [2.0 0.5; 0.5 1.0]
        sir_age = stratify(sir, [:young, :old]; contact=C)
        @test nspecies(sir_age) == 6
        tn = transition_names(sir_age)
        # Interaction transitions should have cross-group versions
        @test :inf_young_young in tn
        @test :inf_young_old in tn
        @test :inf_old_young in tn
        @test :inf_old_old in tn
        # Non-interaction transitions just get group suffix
        @test :rec_young in tn
        @test :rec_old in tn

        # Population should be split equally
        concs = species_concentrations(sir_age)
        total_S = sum(concs[i] for i in 1:length(concs) if species_names(sir_age)[i] in [:S_young, :S_old])
        @test total_S ≈ 990.0
    end

    @testset "ODE lowering" begin
        sir = SIR()
        gen = compile(sir; mode=:ode, frequency_dependent=true, N=:N,
                    params=Dict(:beta => 0.3, :gamma => 0.1, :N => 1000.0))
        @test gen isa OdinModel

        sys = System(gen, (beta=0.3, gamma=0.1, N=1000.0))
        reset!(sys)
        result = simulate(sys, collect(0.0:1.0:100.0))
        r = result[:, 1, end]

        # Population conserved
        @test sum(r) ≈ 1000.0 atol=0.1
        # Epidemic occurred (most went to R)
        @test r[3] > 800
    end

    @testset "Discrete lowering" begin
        sir = SIR()
        gen = compile(sir; mode=:discrete, frequency_dependent=true, N=:N,
                    params=Dict(:beta => 0.5, :gamma => 0.1, :N => 1000.0))
        @test gen isa OdinModel

        sys = System(gen, (beta=0.5, gamma=0.1, N=1000.0);
                                  n_particles=10, dt=0.25, seed=42)
        reset!(sys)
        result = simulate(sys, collect(0.0:1.0:100.0))

        # Check all particles conserve population
        for p in 1:10
            r = result[:, p, end]
            @test sum(r) ≈ 1000.0 atol=0.01
        end
    end

    @testset "SEIR ODE lowering" begin
        seir = SEIR()
        gen = compile(seir; mode=:ode, frequency_dependent=true, N=:N,
                    params=Dict(:beta => 0.5, :sigma => 0.2, :gamma => 0.1, :N => 1000.0))
        sys = System(gen, (beta=0.5, sigma=0.2, gamma=0.1, N=1000.0))
        reset!(sys)
        result = simulate(sys, collect(0.0:1.0:200.0))
        r = result[:, 1, end]

        # Population conserved
        @test sum(r) ≈ 1000.0 atol=0.1
        # Almost all recovered
        @test r[4] > 900
    end

    @testset "Composed model matches direct model" begin
        # Build SIR two ways and check results match
        direct = SIR()
        gen_direct = compile(direct; mode=:ode, frequency_dependent=true, N=:N,
                           params=Dict(:beta => 0.3, :gamma => 0.1, :N => 1000.0))

        inf = EpiNet([:S => 990.0, :I => 10.0],
                     [:inf => ([:S, :I] => [:I, :I], :beta)])
        rec = EpiNet([:I => 10.0, :R => 0.0],
                     [:rec => ([:I] => [:R], :gamma)])
        composed = compose(inf, rec)
        gen_composed = compile(composed; mode=:ode, frequency_dependent=true, N=:N,
                             params=Dict(:beta => 0.3, :gamma => 0.1, :N => 1000.0))

        pars = (beta=0.3, gamma=0.1, N=1000.0)
        times = collect(0.0:1.0:100.0)

        sys_d = System(gen_direct, pars)
        reset!(sys_d)
        r_d = simulate(sys_d, times)[:, 1, :]

        sys_c = System(gen_composed, pars)
        reset!(sys_c)
        r_c = simulate(sys_c, times)[:, 1, :]

        @test r_d ≈ r_c atol=0.01
    end

    @testset "Stratified ODE simulation" begin
        sir = SIR()
        C = [2.0 0.5; 0.5 1.0]
        sir_age = stratify(sir, [:young, :old]; contact=C)
        gen = compile(sir_age; mode=:ode, frequency_dependent=true, N=:N,
                    params=Dict(:beta => 0.3, :gamma => 0.1, :N => 1000.0))
        sys = System(gen, (beta=0.3, gamma=0.1, N=1000.0))
        reset!(sys)
        result = simulate(sys, collect(0.0:1.0:100.0))
        r = result[:, 1, end]

        # Population conserved
        @test sum(r) ≈ 1000.0 atol=0.1

        # Young group hit harder (higher contact rate)
        sn = species_names(sir_age)
        R_young = r[findfirst(==(:R_young), sn)]
        R_old = r[findfirst(==(:R_old), sn)]
        @test R_young > R_old
    end

    @testset "lower_expr returns expressions" begin
        sir = SIR()
        exprs = lower_expr(sir; mode=:ode, frequency_dependent=true, N=:N,
                           params=Dict(:beta => 0.3, :gamma => 0.1, :N => 1000.0))
        @test exprs isa Vector
        @test length(exprs) > 0
        # Should contain parameter, initial, and deriv expressions
        expr_strs = string.(exprs)
        @test any(contains(s, "parameter") for s in expr_strs)
        @test any(contains(s, "initial") for s in expr_strs)
        @test any(contains(s, "deriv") for s in expr_strs)
    end
end
