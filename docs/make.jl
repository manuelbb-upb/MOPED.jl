using MOPEDS
using Documenter

import Literate

DocMeta.setdocmeta!(MOPEDS, :DocTestSetup, :(using MOPEDS); recursive=true)

function src_md(fn)
    in_file = joinpath(@__DIR__, "..", "src", fn)
    out_dir = joinpath(@__DIR__, "src")
    Literate.markdown(in_file, out_dir; flavor=Literate.CommonMarkFlavor(), execute=false)
end

src_md("problem_interface.jl")
src_md("graph.jl")
src_md("bridging_interface.jl")

makedocs(;
    modules=[MOPEDS],
    authors="manuelbb-upb <manuelbb@mail.uni-paderborn.de> and contributors",
    sitename="MOPEDS.jl",
    format=Documenter.HTML(;
        canonical="https://manuelbb-upb.github.io/MOPEDS.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Internals" => [
            "Problem Interface" => "problem_interface.md",
            "Hypergraph" => "graph.md",
            "Bridging" => "bridging_interface.md"
        ]
    ],
)

deploydocs(;
    repo="github.com/manuelbb-upb/MOPEDS.jl",
    devbranch="main",
)
