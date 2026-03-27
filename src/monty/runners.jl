# Runners: strategies for executing MCMC chains.

"""
    AbstractMontyRunner

Abstract type for MCMC chain execution strategies.
"""
abstract type AbstractMontyRunner end

"""
    MontySerialRunner

Run chains sequentially on a single thread.
"""
struct MontySerialRunner <: AbstractMontyRunner end

"""
    monty_runner_serial()

Create a serial runner that executes chains sequentially.
"""
monty_runner_serial() = MontySerialRunner()

"""
    MontyThreadedRunner

Run chains in parallel using Julia threads.
"""
struct MontyThreadedRunner <: AbstractMontyRunner end

"""
    monty_runner_threaded()

Create a threaded runner that executes chains in parallel using Julia threads.
"""
monty_runner_threaded() = MontyThreadedRunner()


"""
    MontySimultaneousRunner

Run all chains simultaneously in lock-step (enables cross-chain interactions).
"""
struct MontySimultaneousRunner <: AbstractMontyRunner end

"""
    monty_runner_simultaneous()

Create a simultaneous runner that advances all chains in lock-step.
"""
monty_runner_simultaneous() = MontySimultaneousRunner()
