using MOPED
using Documenter

DocMeta.setdocmeta!(MOPED, :DocTestSetup, :(using MOPED); recursive=true)

makedocs(;
    modules=[MOPED],
    authors="manuelbb-upb <manuelbb@mail.uni-paderborn.de> and contributors",
    sitename="MOPED.jl",
    format=Documenter.HTML(;
        canonical="https://manuelbb-upb.github.io/MOPED.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/manuelbb-upb/MOPED.jl",
    devbranch="main",
)
