using Test
using Odin

@testset "Categorical Advanced" begin

    @testset "Spatial composition — ring topology" begin
        sir = SIR(; S0=990.0, I0=10.0, R0=0.0)
        patches = [:p1, :p2, :p3]
        C_ring = [1.0 0.1 0.1; 0.1 1.0 0.1; 0.1 0.1 1.0]
        sir_ring = stratify(sir, patches; contact=C_ring)

        # 3 patches × 3 species = 9 species
        @test nspecies(sir_ring) == 9
        sn = species_names(sir_ring)
        @test :S_p1 in sn
        @test :I_p2 in sn
        @test :R_p3 in sn

        # Add migration sub-models
        function make_mig(from, to)
            s_from = Symbol(:S_, from)
            s_to = Symbol(:S_, to)
            t_name = Symbol(:mig_S_, from, :_, to)
            EpiNet([s_from => 0.0, s_to => 0.0],
                   [t_name => ([s_from] => [s_to], :mu)])
        end

        mig_nets = [make_mig(:p1, :p2), make_mig(:p2, :p1),
                    make_mig(:p2, :p3), make_mig(:p3, :p2),
                    make_mig(:p3, :p1), make_mig(:p1, :p3)]

        sir_spatial = compose(sir_ring, mig_nets...)
        # Same 9 species, more transitions
        @test nspecies(sir_spatial) == 9
        @test ntransitions(sir_spatial) > ntransitions(sir_ring)

        # Should contain migration transitions
        tn = transition_names(sir_spatial)
        @test :mig_S_p1_p2 in tn
        @test :mig_S_p3_p1 in tn
    end

    @testset "Spatial composition — ODE simulation conserves population" begin
        sir = SIR(; S0=990.0, I0=10.0, R0=0.0)
        patches = [:p1, :p2, :p3]
        C = [1.0 0.1 0.1; 0.1 1.0 0.1; 0.1 0.1 1.0]
        sir_strat = stratify(sir, patches; contact=C)

        function make_mig(from, to)
            s_from = Symbol(:S_, from)
            s_to = Symbol(:S_, to)
            t_name = Symbol(:mig_S_, from, :_, to)
            EpiNet([s_from => 0.0, s_to => 0.0],
                   [t_name => ([s_from] => [s_to], :mu)])
        end

        mig_nets = [make_mig(:p1, :p2), make_mig(:p2, :p1),
                    make_mig(:p2, :p3), make_mig(:p3, :p2)]
        sir_spatial = compose(sir_strat, mig_nets...)

        gen = compile(sir_spatial; mode=:ode, frequency_dependent=true, N=:N,
                    params=Dict(:beta => 0.3, :gamma => 0.1, :mu => 0.01, :N => 1000.0))
        @test gen isa OdinModel

        sys = System(gen, (beta=0.3, gamma=0.1, mu=0.01, N=1000.0))
        reset!(sys)
        times = collect(0.0:1.0:100.0)
        result = simulate(sys, times)

        # Population conserved at all time points
        for t_idx in 1:length(times)
            total = sum(result[:, 1, t_idx])
            @test total ≈ 1000.0 atol=0.5
        end
    end

    @testset "Spatial composition — star vs ring produces different dynamics" begin
        sir = SIR(; S0=990.0, I0=10.0, R0=0.0)
        patches = [:p1, :p2, :p3]

        C_ring = [1.0 0.1 0.1; 0.1 1.0 0.1; 0.1 0.1 1.0]
        C_star = [1.0 0.2 0.2; 0.2 1.0 0.0; 0.2 0.0 1.0]

        sir_ring = stratify(sir, patches; contact=C_ring)
        sir_star = stratify(sir, patches; contact=C_star)

        gen_ring = compile(sir_ring; mode=:ode, frequency_dependent=true, N=:N,
                         params=Dict(:beta => 0.3, :gamma => 0.1, :N => 1000.0))
        gen_star = compile(sir_star; mode=:ode, frequency_dependent=true, N=:N,
                         params=Dict(:beta => 0.3, :gamma => 0.1, :N => 1000.0))

        pars = (beta=0.3, gamma=0.1, N=1000.0)
        times = collect(0.0:1.0:100.0)

        sys_r = System(gen_ring, pars)
        reset!(sys_r)
        r_ring = simulate(sys_r, times)

        sys_s = System(gen_star, pars)
        reset!(sys_s)
        r_star = simulate(sys_s, times)

        # Both topologies should differ in dynamics
        @test !(r_ring[:, 1, :] ≈ r_star[:, 1, :])

        # But both conserve population
        @test sum(r_ring[:, 1, end]) ≈ 1000.0 atol=0.5
        @test sum(r_star[:, 1, end]) ≈ 1000.0 atol=0.5
    end

    @testset "Stratification — dimensions match" begin
        sir = SIR()
        groups2 = [:young, :old]
        groups3 = [:child, :adult, :elder]
        groups4 = [:g1, :g2, :g3, :g4]

        sir2 = stratify(sir, groups2)
        sir3 = stratify(sir, groups3)
        sir4 = stratify(sir, groups4)

        @test nspecies(sir2) == 3 * 2  # 3 base species × 2 groups
        @test nspecies(sir3) == 3 * 3
        @test nspecies(sir4) == 3 * 4

        # SEIR has 4 species
        seir = SEIR()
        seir3 = stratify(seir, groups3)
        @test nspecies(seir3) == 4 * 3
    end

    @testset "Stratification — assortative mixing equals independent" begin
        sir = SIR(; S0=990.0, I0=10.0, R0=0.0)
        groups = [:a, :b]

        # Assortative: identity contact matrix
        C_id = [1.0 0.0; 0.0 1.0]
        sir_assort = stratify(sir, groups; contact=C_id)

        gen = compile(sir_assort; mode=:ode, frequency_dependent=true, N=:N,
                    params=Dict(:beta => 0.3, :gamma => 0.1, :N => 1000.0))
        sys = System(gen, (beta=0.3, gamma=0.1, N=1000.0))
        reset!(sys)
        result = simulate(sys, collect(0.0:1.0:100.0))

        sn = species_names(sir_assort)
        I_a = findfirst(==(:I_a), sn)
        I_b = findfirst(==(:I_b), sn)

        # With equal initial conditions and identity contact, groups are identical
        @test result[I_a, 1, :] ≈ result[I_b, 1, :] atol=0.01
    end

    @testset "Stratification — contact matrix affects dynamics" begin
        sir = SIR(; S0=990.0, I0=10.0, R0=0.0)
        groups = [:young, :old]

        # Young has much higher contact
        C = [3.0 0.5; 0.5 1.0]
        sir_age = stratify(sir, groups; contact=C)

        gen = compile(sir_age; mode=:ode, frequency_dependent=true, N=:N,
                    params=Dict(:beta => 0.2, :gamma => 0.1, :N => 1000.0))
        sys = System(gen, (beta=0.2, gamma=0.1, N=1000.0))
        reset!(sys)
        times = collect(0.0:1.0:200.0)
        result = simulate(sys, times)

        sn = species_names(sir_age)
        R_young = result[findfirst(==(:R_young), sn), 1, end]
        R_old = result[findfirst(==(:R_old), sn), 1, end]

        # Young group should have higher attack rate
        @test R_young > R_old

        # Population conserved
        @test sum(result[:, 1, end]) ≈ 1000.0 atol=0.5
    end

    @testset "Stratification with vaccination composition" begin
        sir = SIR(; S0=990.0, I0=10.0, R0=0.0)
        groups = [:child, :adult, :elder]
        C = [3.0 1.0 0.3; 1.0 2.0 0.5; 0.3 0.5 1.5]
        sir_strat = stratify(sir, groups; contact=C)

        # Add vaccination for children
        vax = EpiNet([:S_child => 0.0, :V_child => 0.0],
                     [:vax_child => ([:S_child] => [:V_child], :nu)])
        sir_vax = compose(sir_strat, vax)

        # Check species include V_child
        sn = species_names(sir_vax)
        @test :V_child in sn
        @test :S_child in sn
        @test nspecies(sir_vax) == 10  # 9 from stratification + V_child

        # Simulate with vaccination
        gen = compile(sir_vax; mode=:ode, frequency_dependent=true, N=:N,
                    params=Dict(:beta => 0.15, :gamma => 0.1, :nu => 0.01, :N => 1000.0))
        sys = System(gen, (beta=0.15, gamma=0.1, nu=0.01, N=1000.0))
        reset!(sys)
        result = simulate(sys, collect(0.0:1.0:200.0))

        # Vaccinated compartment should have people
        V_idx = findfirst(==(:V_child), sn)
        @test result[V_idx, 1, end] > 0
    end

    @testset "Multi-pathogen — disjoint composition" begin
        flu = EpiNet(
            [:S_flu => 495.0, :I_flu => 5.0, :R_flu => 0.0],
            [:inf_flu => ([:S_flu, :I_flu] => [:I_flu, :I_flu], :beta_flu),
             :rec_flu => ([:I_flu] => [:R_flu], :gamma_flu)]
        )
        chl = EpiNet(
            [:S_chl => 495.0, :I_chl => 5.0],
            [:inf_chl => ([:S_chl, :I_chl] => [:I_chl, :I_chl], :beta_chl),
             :rec_chl => ([:I_chl] => [:S_chl], :gamma_chl)]
        )

        combined = compose(flu, chl)
        @test nspecies(combined) == 5  # S_flu, I_flu, R_flu, S_chl, I_chl
        @test ntransitions(combined) == 4
        @test Set(species_names(combined)) == Set([:S_flu, :I_flu, :R_flu, :S_chl, :I_chl])

        gen = compile(combined; mode=:ode, frequency_dependent=true, N=:N,
                    params=Dict(:beta_flu => 0.4, :gamma_flu => 0.15,
                                :beta_chl => 0.3, :gamma_chl => 0.05, :N => 500.0))
        sys = System(gen, (beta_flu=0.4, gamma_flu=0.15,
                                       beta_chl=0.3, gamma_chl=0.05, N=500.0))
        reset!(sys)
        result = simulate(sys, collect(0.0:1.0:200.0))

        sn = species_names(combined)

        # Flu sub-population conserved
        S_flu = result[findfirst(==(:S_flu), sn), 1, end]
        I_flu = result[findfirst(==(:I_flu), sn), 1, end]
        R_flu = result[findfirst(==(:R_flu), sn), 1, end]
        @test S_flu + I_flu + R_flu ≈ 500.0 atol=0.5

        # Chlamydia sub-population conserved
        S_chl = result[findfirst(==(:S_chl), sn), 1, end]
        I_chl = result[findfirst(==(:I_chl), sn), 1, end]
        @test S_chl + I_chl ≈ 500.0 atol=0.5

        # Flu epidemic occurred
        @test R_flu > 400
    end

    @testset "Multi-pathogen — joint state model" begin
        two_pathogen = @odin begin
            beta_flu = parameter(0.4)
            gamma_flu = parameter(0.15)
            beta_chl = parameter(0.3)
            gamma_chl = parameter(0.05)
            N = parameter(1000.0)

            total_I_flu = IS + II
            total_I_chl = SI + II + RI

            foi_flu = beta_flu * total_I_flu / N
            foi_chl = beta_chl * total_I_chl / N

            deriv(SS) = -foi_flu * SS - foi_chl * SS + gamma_chl * SI
            deriv(IS) = foi_flu * SS - gamma_flu * IS - foi_chl * IS + gamma_chl * II
            deriv(SI) = foi_chl * SS - foi_flu * SI - gamma_chl * SI
            deriv(II) = foi_flu * SI + foi_chl * IS - gamma_flu * II - gamma_chl * II
            deriv(RS) = gamma_flu * IS - foi_chl * RS + gamma_chl * RI
            deriv(RI) = gamma_flu * II + foi_chl * RS - gamma_chl * RI

            initial(SS) = 880.0
            initial(IS) = 5.0
            initial(SI) = 5.0
            initial(II) = 0.0
            initial(RS) = 0.0
            initial(RI) = 0.0
        end

        pars = (beta_flu=0.4, gamma_flu=0.15, beta_chl=0.3, gamma_chl=0.05, N=1000.0)
        sys = System(two_pathogen, pars)
        reset!(sys)
        result = simulate(sys, collect(0.0:1.0:300.0))

        # Population conserved (total = 890)
        total = sum(result[:, 1, end])
        @test total ≈ 890.0 atol=0.5

        # All compartments non-negative
        @test all(result[:, 1, :] .>= -0.01)

        # Flu epidemic occurred: RS + RI should be large
        @test result[5, 1, end] + result[6, 1, end] > 700
    end

    @testset "Composed model matches manually written spatial model" begin
        # Build 2-patch SIR with composition
        sir = SIR(; S0=990.0, I0=10.0, R0=0.0)
        patches = [:a, :b]
        C = [1.0 0.0; 0.0 1.0]  # no cross-patch infection (simple case)
        sir_2p = stratify(sir, patches; contact=C)

        # Add symmetric migration
        mig_ab = EpiNet([:S_a => 0.0, :S_b => 0.0],
                        [:mig_a_b => ([:S_a] => [:S_b], :mu)])
        mig_ba = EpiNet([:S_a => 0.0, :S_b => 0.0],
                        [:mig_b_a => ([:S_b] => [:S_a], :mu)])
        sir_composed = compose(sir_2p, mig_ab, mig_ba)

        gen_c = compile(sir_composed; mode=:ode, frequency_dependent=true, N=:N,
                      params=Dict(:beta => 0.3, :gamma => 0.1, :mu => 0.01, :N => 1000.0))

        # Manual equivalent
        manual_2p = @odin begin
            beta = parameter(0.3)
            gamma = parameter(0.1)
            mu = parameter(0.01)
            N = parameter(1000.0)

            flow_inf_a = beta * S_a * I_a / N
            flow_rec_a = gamma * I_a
            deriv(S_a) = -flow_inf_a - mu * S_a + mu * S_b
            deriv(I_a) = flow_inf_a - flow_rec_a
            deriv(R_a) = flow_rec_a

            flow_inf_b = beta * S_b * I_b / N
            flow_rec_b = gamma * I_b
            deriv(S_b) = -flow_inf_b + mu * S_a - mu * S_b
            deriv(I_b) = flow_inf_b - flow_rec_b
            deriv(R_b) = flow_rec_b

            initial(S_a) = 495.0
            initial(I_a) = 5.0
            initial(R_a) = 0.0
            initial(S_b) = 495.0
            initial(I_b) = 5.0
            initial(R_b) = 0.0
        end

        pars = (beta=0.3, gamma=0.1, mu=0.01, N=1000.0)
        times = collect(0.0:1.0:100.0)

        sys_c = System(gen_c, pars)
        reset!(sys_c)
        r_c = simulate(sys_c, times)

        sys_m = System(manual_2p, pars)
        reset!(sys_m)
        r_m = simulate(sys_m, times)

        # Both should conserve population
        @test sum(r_c[:, 1, end]) ≈ 1000.0 atol=0.5
        @test sum(r_m[:, 1, end]) ≈ 1000.0 atol=0.5

        # Final epidemic sizes should be similar (both should be hit)
        # The composed model species ordering may differ from manual,
        # so compare total recovered
        total_R_c = sum(r_c[i, 1, end] for i in 1:nspecies(sir_composed)
                       if startswith(string(species_names(sir_composed)[i]), "R_"))
        total_R_m = r_m[3, 1, end] + r_m[6, 1, end]
        @test total_R_c ≈ total_R_m atol=5.0
    end

end
