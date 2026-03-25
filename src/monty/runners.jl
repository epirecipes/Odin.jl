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
