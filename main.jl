using Statistics
using Printf

using Images
using Noise

include("hopfield.jl")

using .HopfieldNetwork

const base_d = "data/EnglishHnd/English/Hnd/Img/"
const pre_d = "Sample"

function get_img_files(s_char::AbstractChar)
    n = s_char <= '9' ? s_char - '0' + 1 : s_char - 'A' + 11
    readdir(joinpath(base_d, pre_d*(@sprintf("%03d", n))), join=true)
end

function get_img_hnd(
        out_t::typeof(AbstractFloat),
        s_char::AbstractChar,
        (h, w)::Tuple{Integer, Integer},
)::Vector{Matrix{out_t}}
    Matrix{out_t}[out_t.(Gray.(imresize(load(f), (h, w)))) for f ∈ get_img_files(s_char)]
end

function get_data(chars::StepRange{Char, Int}, (h, w)::Tuple{Integer, Integer}; wnoise::Bool=false)::Matrix{Float64}
    rawdata = Matrix{Float64}[]
    for c in chars
        imgs = get_img_hnd(Float64, c, (h, w))
        best_img = imgs[argmin(mean.(imgs))]
        if wnoise
            best_img = salt_pepper(best_img)
        end
        push!(rawdata, best_img)
    end
    data = Vector{Float64}[]
    for d in rawdata
        push!(data, vcat(d...))
    end
    data = collect(hcat(data...)')

    rep(x) = iszero(x) ? 1.0 : -1.0
    data = rep.(data)
    data
end

function run_test(chars::StepRange{Char, Int}, (h, w)::Tuple{Integer, Integer}; wnoise::Bool=false)::AbstractFloat
    traindata = get_data(chars, (h, w))
    testdata = get_data(chars, (h, w), wnoise=wnoise)
    hn = SimpleHN{Float64}(traindata, testdata)
    1 - predict_error(hn)
end

function hamming_dis(chars::StepRange{Char, Int}, (h, w)::Tuple{Integer, Integer})::Matrix{Float64}
    data = get_data(chars, (h, w))
    res = zeros(size(data, 1), size(data, 1))
    for (i1, x1) ∈ enumerate(eachrow(data)), (i2, x2) ∈ enumerate(eachrow(data))
        res[i1, i2] = sum(x1 .!= x2)
    end
    res
end

let quz = "question-1-2"
    @show run_test('0':'4', (10, 10))
    @show run_test('A':'E', (10, 10))
    println("$(quz) done.")
end

let quz = "question-1-3"
    println("0--4")
    h0_4 = hamming_dis('0':'4', (10, 10))
    display(h0_4)
    println()
    @show mean(h0_4)

    println("A--E")
    hA_E = hamming_dis('A':'E', (10, 10))
    display(hA_E)
    println()
    @show mean(hA_E)
    println("$(quz) done.")
end

let quz = "question-1-4"
    @show run_test('0':'4', (10, 10), wnoise=true)
    @show run_test('A':'E', (10, 10), wnoise=true)
    println("$(quz) done.")
end

let quz = "question-2"
    @show run_test('0':'4', (30, 30))
    @show run_test('A':'E', (30, 30))
    println("$(quz) done.")
end

let quz = "question-3"
    @show run_test('0':'9', (30, 30))
    @show run_test('A':'O', (30, 30))
    println("$(quz) done.")
end