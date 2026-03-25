# ACSet schemas for epidemiological networks using Catlab.
#
# Defines labelled Petri net schemas where species represent compartments
# (e.g. S, I, R) and transitions represent epidemiological processes
# (e.g. infection, recovery).

# Note: Catlab macros and functions imported in main Odin.jl module

# Labelled reaction net schema (species + transitions + arcs + attributes)
@present SchLabelledReactionNet(FreeSchema) begin
    T::Ob     # Transitions (epidemiological processes)
    S::Ob     # Species (compartments)
    I::Ob     # Input arcs (transition consumes species)
    O::Ob     # Output arcs (transition produces species)
    it::Hom(I, T)   # Input arc → transition
    is::Hom(I, S)   # Input arc → species
    ot::Hom(O, T)   # Output arc → transition
    os::Hom(O, S)   # Output arc → species
    Name::AttrType
    Rate::AttrType
    Conc::AttrType
    tname::Attr(T, Name)         # Transition name (e.g. :inf, :rec)
    rate::Attr(T, Rate)          # Rate expression (Symbol, Number, or Expr)
    sname::Attr(S, Name)         # Species name (e.g. :S, :I, :R)
    concentration::Attr(S, Conc) # Initial concentration
end

@acset_type EpiNetACSet(SchLabelledReactionNet, index=[:it, :is, :ot, :os])
