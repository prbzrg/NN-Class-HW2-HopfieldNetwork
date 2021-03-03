module HopfieldNetwork

    using Statistics
    using LinearAlgebra

    export
        SimpleHN,
        predict_error

    abstract type NeuralNetwork end
    abstract type RecurrentNN <: NeuralNetwork end
    abstract type HopfieldN <: RecurrentNN end

    mutable struct SimpleHN{T} <: HopfieldN where T <: AbstractFloat
        traindata::Matrix{T}
        testdata::Matrix{T}

        n_epochs::Integer

        w::Matrix{T}

        function SimpleHN{T}(
            traindata::Matrix{T},
            testdata::Matrix{T},
            ;
            n_epochs::Integer=1000,
        ) where T <: AbstractFloat
            w = (traindata' * traindata) / 4
            w -= Diagonal(w)
            new(traindata, testdata, n_epochs, w)
        end
    end

    function predict_error(hn::HopfieldN)::AbstractFloat
        res = Vector[]
        for s ∈ eachrow(hn.testdata)
            s = hcat(s)
            in_d = collect(s')
            out_d = sign.(in_d * hn.w)
            c = 1
            while in_d != out_d && c <= hn.n_epochs
                in_d = out_d
                out_d = sign.(in_d * hn.w)
                c += 1
            end
            push!(res, vcat(out_d...))
        end
        res = collect(hcat(res...)')
        mean(abs.(hn.traindata - res))
    end
end