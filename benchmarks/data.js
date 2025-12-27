window.BENCHMARK_DATA = {
  "lastUpdate": 1766807845724,
  "repoUrl": "https://github.com/EnzymeAD/Reactant.jl",
  "entries": {
    "Reactant.jl Benchmarks": [
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "e6447e9959049a05f1502b5ed33ece761e454d31",
          "message": "Regenerate MLIR Bindings (#2025)\n\nCo-authored-by: enzyme-ci-bot[bot] <78882869+enzyme-ci-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-12-25T01:27:41Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/e6447e9959049a05f1502b5ed33ece761e454d31"
        },
        "date": 1766634849725,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.1631182753,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.0015008307999999997,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.0046466663,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.17723769309999998,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAll",
            "value": 0.0048167624,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004553473,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CPU/StructuredTensors",
            "value": 3.7214361811200005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.1747800876,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.17395007339999996,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/XLA",
            "value": 0.16191282499999998,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.17375150870000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisablePad",
            "value": 0.07714771220000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CPU/Julia",
            "value": 5.133643739,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisablePad",
            "value": 0.0013673094999999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.173457718,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CPU/Julia",
            "value": 0.466674805,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAll",
            "value": 0.17485067329999998,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0044580386,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.1619054829,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.0045926382000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.004500671899999999,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CPU/Default",
            "value": 0.050233903079999996,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CPU/Default",
            "value": 2.2960830332400004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableTransposeReshape",
            "value": 0.07593766449999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004542084,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/XLA",
            "value": 0.0020048578000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/XLA",
            "value": 0.0049879029000000005,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CPU/StructuredTensors",
            "value": 0.02605822912,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CPU/Julia",
            "value": 0.091548658,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.162858478,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CPU/Default",
            "value": 0.32269379528,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0045431294,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.16486414630000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableTransposeReshape",
            "value": 0.0015008307999999997,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CPU/Default",
            "value": 0.013868821239999997,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CPU/Default",
            "value": 0.008000193479999999,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CPU/Julia",
            "value": 0.019337691,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.16490664840000002,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CPU/StructuredTensors",
            "value": 0.10010216936,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.0050667952,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.004551710800000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGatherPad",
            "value": 0.0766407093,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.0045845989,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAll",
            "value": 0.004520243799999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.1760389165,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CPU/Julia",
            "value": 0.005542915000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGather",
            "value": 0.0759745472,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CPU/StructuredTensors",
            "value": 0.5801728186799999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGatherPad",
            "value": 0.0013452944999999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.004527336800000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0043623675,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CPU/StructuredTensors",
            "value": 0.010222107760000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGather",
            "value": 0.0013738837,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAll",
            "value": 0.17289024479999998,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0044040563,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/XLA",
            "value": 0.07772809759999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.0765384264,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.17431939180000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.17358758709999997,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.0019096292,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.0006017568000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/XLA",
            "value": 0.0007452968,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.0071392249,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.0006098245,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CUDA/Default",
            "value": 0.04024473075999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.0031518956000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.0019036714,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007233043600000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.0031623157,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/XLA",
            "value": 0.00022877890000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0031411642,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.0031166611,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisablePad",
            "value": 0.0010878752,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007219302899999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0072059086,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0071438527,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.0010950462000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAll",
            "value": 0.0005957696,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CUDA/Default",
            "value": 0.0008220574800000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CUDA/StructuredTensors",
            "value": 0.0014741738,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableTransposeReshape",
            "value": 0.00021867639999999998,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.0028422435,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.002840433,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.0006120804,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/XLA",
            "value": 0.011917813,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0031559069000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.0071494563999999995,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGather",
            "value": 0.0002331533,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001123162,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.0031543466,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/XLA",
            "value": 0.0031063466000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007210588499999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003304207,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.0027462986,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.007129544,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007149625,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CUDA/StructuredTensors",
            "value": 0.00724098532,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisablePad",
            "value": 0.000248555,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0033272298,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0005840575000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0005938816999999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.0071359119,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0005975711999999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007136421300000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/XLA",
            "value": 0.0033419316000000004,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CUDA/Default",
            "value": 0.007045649040000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/XLA",
            "value": 0.007259772099999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.0019062032999999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0032965144,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.0006049244,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.0033750546,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.007135305,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CUDA/StructuredTensors",
            "value": 0.00239302196,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CUDA/Default",
            "value": 0.0015108134,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0005933663000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/XLA",
            "value": 0.0011370460999999999,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CUDA/StructuredTensors",
            "value": 0.02274414568,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.0028274216000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/XLA",
            "value": 0.0019307954999999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.007143876999999998,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0031441756,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.0028877185000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0005976578,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001078326,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CUDA/StructuredTensors",
            "value": 0.11064856216,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0071443405000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.0005911756,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.0005938593000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.0031259516000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.0031964054000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.000602397,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.0019077128999999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.0010904478,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGatherPad",
            "value": 0.0002252618,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.0019098441,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.0112379345,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CUDA/Default",
            "value": 0.00058065112,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0006105632,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.0002293196,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0031494743000000007,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0031199645,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.0072070412,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0006105602999999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004179129375,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisablePad",
            "value": 0.00092897425,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.003093510375,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/XLA",
            "value": 0.002020014875,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.000027264375,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000027185125,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0030858635,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.003093168,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/TPU/StructuredTensors",
            "value": 0.0207960866,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.00417976125,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000027328625,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.0000060418750000000005,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/TPU/Default",
            "value": 0.020862594900000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.0041794245,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000929653875,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0029888841249999997,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.000027244250000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.000027204,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000217914625,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.002985304625,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.002987499375,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004179710875,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000953035,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.0030930124999999998,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/TPU/Default",
            "value": 0.000038255049999999996,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/XLA",
            "value": 0.000587868375,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.0016974346250000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.0009295815,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGatherPad",
            "value": 0.000006067375,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027264625,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.00417964975,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/TPU/Default",
            "value": 0.00020746945,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGather",
            "value": 0.000006054,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.0009290412499999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003100048375,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisablePad",
            "value": 0.00021769375000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027272,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisablePad",
            "value": 0.00000606175,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/TPU/StructuredTensors",
            "value": 0.00003905985,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/XLA",
            "value": 0.000027369875,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.004178525000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.0030932495,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.000027222375000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004180741375,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00418024625,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.0041797075,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.0030854015,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004179712125,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004179633875,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/XLA",
            "value": 0.00086715275,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.00021544437499999998,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.00021773700000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003093658375,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.0009529755000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/XLA",
            "value": 0.004054636125,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/XLA",
            "value": 0.00113816,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableTransposeReshape",
            "value": 0.000006133375,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAll",
            "value": 0.0030937975,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/TPU/StructuredTensors",
            "value": 0.00020819085000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/XLA",
            "value": 0.0000062835,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.0009529648749999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.004180098375,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027201875,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.004180499125,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.00418006525,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000929606,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000217693125,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/TPU/StructuredTensors",
            "value": 0.0000181592,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003093609375,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000027243375000000002,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/TPU/StructuredTensors",
            "value": 0.0015397231,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/TPU/Default",
            "value": 0.0015230170000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000027263625000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAll",
            "value": 0.000027331375,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAll",
            "value": 0.000027217499999999998,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.0009852655,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.0030853053750000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0030851259999999997,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/TPU/Default",
            "value": 0.0000173675,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004179685875,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisablePad",
            "value": 0.000953032,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/XLA",
            "value": 0.002959022125,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.000027226874999999997,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.000027241625000000002,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Avik Pal",
            "username": "avik-pal",
            "email": "avikpal@mit.edu"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "316679af35b0a7ff2c1b31ffe80895b2fb0a4e5a",
          "message": "perf: standardize naming (#2026)",
          "timestamp": "2025-12-25T06:24:06Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/316679af35b0a7ff2c1b31ffe80895b2fb0a4e5a"
        },
        "date": 1766676834619,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAll",
            "value": 0.006024842799999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAll",
            "value": 0.005821686500000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005869778300000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0060257737000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.006328668200000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.005975246800000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CPU/Default",
            "value": 0.3694223178,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.21143358130000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.0058574718,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CPU/Julia",
            "value": 0.6206285340000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CPU/Julia",
            "value": 0.007260202,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.005997547800000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CPU/Default",
            "value": 0.06311101572,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CPU/StructuredTensors",
            "value": 0.130487122,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CPU/StructuredTensors",
            "value": 4.052520894320001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001975256,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGatherPad",
            "value": 0.0017192546,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CPU/Default",
            "value": 0.01855256352,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.005686961,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/XLA",
            "value": 0.0023556719,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisablePad",
            "value": 0.09009742859999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.19891665629999997,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGather",
            "value": 0.0017567741,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.005808905799999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.2122337946,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.19819098429999998,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.005894324900000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.006076251600000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableTransposeReshape",
            "value": 0.0017265334,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.21031309110000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.19818227740000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisablePad",
            "value": 0.0017192546,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CPU/StructuredTensors",
            "value": 0.01497615892,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.005729513100000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGatherPad",
            "value": 0.0900414034,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAll",
            "value": 0.21233292750000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/XLA",
            "value": 0.19661778439999997,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.21192997620000006,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/XLA",
            "value": 0.08916173289999998,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CPU/StructuredTensors",
            "value": 0.6834012007999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.2119480186,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CPU/Default",
            "value": 2.45450974136,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CPU/Default",
            "value": 0.012050517360000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.1958772503,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CPU/Julia",
            "value": 0.024760554,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableTransposeReshape",
            "value": 0.0900820761,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAll",
            "value": 0.21135637889999997,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.211324503,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CPU/Julia",
            "value": 0.12757390500000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.21190348709999998,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CPU/StructuredTensors",
            "value": 0.03574973331999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.1947623663,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.21264549149999995,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGather",
            "value": 0.0890453086,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.006070237700000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.08612396630000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/XLA",
            "value": 0.0060118013,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CPU/Julia",
            "value": 5.958441445,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005702519600000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.0010449017,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0005829561000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.0010796177000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0030949503,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.0028768322000000007,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.0005712921000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007108964999999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.0071312117999999996,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.0018917873999999997,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0032718931,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CUDA/StructuredTensors",
            "value": 0.0014751317600000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.0032141629,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0005663644,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.0032292715,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisablePad",
            "value": 0.0010387284000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.0018943276,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.0030542555,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030833393999999993,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030896035000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0005836847000000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CUDA/StructuredTensors",
            "value": 0.022968716160000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0032301723999999996,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.0071031284,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.0027978789999999996,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.0027781343999999995,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/XLA",
            "value": 0.0116499953,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.0005892086999999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.0071010512999999985,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.0030667165,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.0002190911,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/XLA",
            "value": 0.0032704222,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.0071180865,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CUDA/Default",
            "value": 0.00885404928,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.0018942668000000002,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CUDA/StructuredTensors",
            "value": 0.0023758782,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007171145699999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007175042500000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisablePad",
            "value": 0.0002172926,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.0010552884,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.0031119705000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0005748183,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.0005795608000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0071066155999999995,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.0071035978,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.007106820400000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007111567499999999,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CUDA/Default",
            "value": 0.0019230845600000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/XLA",
            "value": 0.007228472100000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/XLA",
            "value": 0.0002263354,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007190655600000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableTransposeReshape",
            "value": 0.00021140459999999996,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0005690167999999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAll",
            "value": 0.0005817623,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0005671447,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.0030950426,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/XLA",
            "value": 0.0010916462000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.0019068083000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007188374000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0005733822000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007174992399999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/XLA",
            "value": 0.0019150535999999999,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.0026594043000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/XLA",
            "value": 0.0007165272,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CUDA/Default",
            "value": 0.00089074872,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CUDA/Default",
            "value": 0.040209069199999996,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.00059836,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0030806908,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/XLA",
            "value": 0.002991485,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.0018931045,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.0030850325000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.0005744183,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CUDA/StructuredTensors",
            "value": 0.11035152344,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.007103659799999999,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.0108591065,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CUDA/StructuredTensors",
            "value": 0.0071895860799999975,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.0010549846,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0030913305,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CUDA/Default",
            "value": 0.0005707285599999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.0005919641999999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.0005762907,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGatherPad",
            "value": 0.0002215134,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGather",
            "value": 0.00024242209999999998,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.0031008262,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.0027560256000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00298500225,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.002987758875,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/XLA",
            "value": 0.000586798,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGatherPad",
            "value": 0.00000605325,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.000217740875,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/XLA",
            "value": 0.000006287875,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003093243625,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/TPU/Default",
            "value": 0.0000173466,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGather",
            "value": 0.0009531035,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/TPU/StructuredTensors",
            "value": 0.0000390773,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.004180436,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.0030929729999999997,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableTransposeReshape",
            "value": 0.00000606825,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003093030375,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.0009851456249999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/XLA",
            "value": 0.001138131625,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000027239750000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000928879,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00417890375,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/TPU/Default",
            "value": 0.00003824855,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGather",
            "value": 0.000006120625,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.004180026125,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000952999375,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0029881655,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/XLA",
            "value": 0.004054005125,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/XLA",
            "value": 0.002020066875,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004181079125,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.00000605375,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00418018525,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/TPU/Default",
            "value": 0.00152312685,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.000027198875,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.00417997625,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.000027245874999999998,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.000027309,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisablePad",
            "value": 0.0009290801249999999,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.00021781075000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085472125,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.000027176625,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.0009295036249999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00002725275,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAll",
            "value": 0.000027238749999999997,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000027207625,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.0009294749999999999,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.00021765087500000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAll",
            "value": 0.00309338,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.0009528857499999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00418094075,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000027227000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.000027301624999999998,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/TPU/StructuredTensors",
            "value": 0.0208082734,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0030852106250000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/XLA",
            "value": 0.0008670683749999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/XLA",
            "value": 0.00295970475,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004180961625,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00418016825,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.00309343425,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/TPU/StructuredTensors",
            "value": 0.00153962075,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.0031003786249999997,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.004181163999999999,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/TPU/Default",
            "value": 0.00020748845,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929343625,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/TPU/StructuredTensors",
            "value": 0.00001813345,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003093500875,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisablePad",
            "value": 0.0000060514999999999996,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004179749375,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027238000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.000027254375000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.0030932215,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004179849,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisablePad",
            "value": 0.00095323175,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.00417952875,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.00169749475,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/TPU/StructuredTensors",
            "value": 0.0002081686,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAll",
            "value": 0.00002724625,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027223375000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.00418050725,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000215743125,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisablePad",
            "value": 0.00021774124999999998,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/TPU/Default",
            "value": 0.020837139699999997,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.003085365625,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004180836375,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/XLA",
            "value": 0.000027348875,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.00002728475,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.003085401125,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "William Moses",
            "username": "wsmoses",
            "email": "gh@wsmoses.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ea638a0477ddd699a5df346a8793ee42f0f96d85",
          "message": "Update ENZYMEXLA_COMMIT hash in WORKSPACE",
          "timestamp": "2025-12-26T03:04:34Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/ea638a0477ddd699a5df346a8793ee42f0f96d85"
        },
        "date": 1766723141564,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAll",
            "value": 0.0047648175,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAll",
            "value": 0.004720571999999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004739968000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0046525269000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0048706419,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CPU/Default",
            "value": 0.013788253200000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.0051722630999999995,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.1828920215,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CPU/Julia",
            "value": 5.369871199,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.004914326,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CPU/Default",
            "value": 0.05128447744,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.0049539862,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CPU/Default",
            "value": 0.32934130619999996,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CPU/Default",
            "value": 0.00847228852,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.0015103056,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGatherPad",
            "value": 0.0013867983,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0048968457,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CPU/StructuredTensors",
            "value": 0.012156836840000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/XLA",
            "value": 0.0019540072,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisablePad",
            "value": 0.08433979630000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.17596275700000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGather",
            "value": 0.0014826988999999998,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004693469,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.1857915789,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.17564365149999997,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CPU/Julia",
            "value": 0.019718587000000003,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CPU/Julia",
            "value": 0.00590449,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.004846024700000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.005064203000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableTransposeReshape",
            "value": 0.0014321285000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.1863723018,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CPU/StructuredTensors",
            "value": 0.59673298284,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.17407636359999998,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisablePad",
            "value": 0.0013867983,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CPU/Julia",
            "value": 0.09687348100000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.004795109600000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGatherPad",
            "value": 0.08198882880000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CPU/Julia",
            "value": 0.538865346,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAll",
            "value": 0.1842255494,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CPU/StructuredTensors",
            "value": 3.7749595590800005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/XLA",
            "value": 0.1755046017,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.18792647540000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/XLA",
            "value": 0.08375994679999998,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.186797251,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.1718312777,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableTransposeReshape",
            "value": 0.0823873911,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAll",
            "value": 0.18720884919999997,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.18827216419999998,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.1885571677,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CPU/StructuredTensors",
            "value": 0.11188580072000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.1725821921,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.18639907619999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGather",
            "value": 0.08347305740000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.0047625982999999995,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.082491142,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/XLA",
            "value": 0.0047172185,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CPU/StructuredTensors",
            "value": 0.027830190520000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0048309333,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CPU/Default",
            "value": 2.3681386208400004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.0010965208,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0006200747999999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.0011128209,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0031301989,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.0028512796999999998,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.0006175054,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0071074078999999995,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.007116439800000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.0019068155,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0033217375,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.0031364793000000003,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CUDA/StructuredTensors",
            "value": 0.0014800004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006030600999999998,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.0032777922,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisablePad",
            "value": 0.0010814198,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.0019176706000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.0030978278,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0031320432999999994,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0031063448,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0006158286,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0032803355,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.007111417200000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.0028561925,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.0028809055999999998,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/XLA",
            "value": 0.011124480499999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.0006188683999999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.0071023606000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.0031091321,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.0002311875,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/XLA",
            "value": 0.0033091516999999996,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007114067900000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.0019125281999999998,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.0071803408,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0071815348,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisablePad",
            "value": 0.00023170920000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.0010838215,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.0032998574000000004,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CUDA/Default",
            "value": 0.0019185292800000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0006101552999999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.0006557385,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0071068604,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007121154700000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.007104307799999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0071205143,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/XLA",
            "value": 0.007216101399999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/XLA",
            "value": 0.0002322209,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0071835735,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableTransposeReshape",
            "value": 0.00022105969999999998,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0006125073,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAll",
            "value": 0.0006188589,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0006067948,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CUDA/StructuredTensors",
            "value": 0.1106216538,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CUDA/StructuredTensors",
            "value": 0.00238447084,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.0031359261,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/XLA",
            "value": 0.0011656889,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.0019024883999999998,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0071821805,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0006122448,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007176418800000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CUDA/Default",
            "value": 0.04021056943999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/XLA",
            "value": 0.0019343765999999998,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CUDA/StructuredTensors",
            "value": 0.0230508122,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.0027398999999999995,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/XLA",
            "value": 0.0007352013999999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.0006495371,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CUDA/StructuredTensors",
            "value": 0.007238643320000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0031037981999999997,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/XLA",
            "value": 0.0030480249,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.0019053311,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.0031438466,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.0006087306,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CUDA/Default",
            "value": 0.0005883593199999999,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CUDA/Default",
            "value": 0.00845520548,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.007113657399999999,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.0110434569,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CUDA/Default",
            "value": 0.00082661892,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.0010760145000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0031341786999999994,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.0006209131000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.0006175017,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGatherPad",
            "value": 0.00023936990000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGather",
            "value": 0.0002279108,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.0031079901,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.0028544640999999997,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.002986910875,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/TPU/StructuredTensors",
            "value": 0.02115795855,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.002984138125,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/XLA",
            "value": 0.00058704525,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGatherPad",
            "value": 0.000006104875,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.00021787275,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/XLA",
            "value": 0.000006244874999999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003100398875,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGather",
            "value": 0.00095306725,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.0041820785,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.0031004861250000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableTransposeReshape",
            "value": 0.0000061355,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0031003719999999997,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.0009851747500000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/XLA",
            "value": 0.00113838975,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000027493124999999998,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000929261625,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/TPU/Default",
            "value": 0.00152360005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004179945375,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGather",
            "value": 0.0000060995,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.0041822032500000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.0009530075,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.002985505875,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/XLA",
            "value": 0.00405384875,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/XLA",
            "value": 0.002020982125,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004181626125,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006185875,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004181098124999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.0000274455,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.004181318125,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.000027398374999999997,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.000027444125,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000929444125,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000217818625,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085188875,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/TPU/Default",
            "value": 0.0000174341,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.000027451125,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000929639,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00002735975,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/TPU/StructuredTensors",
            "value": 0.00154054255,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAll",
            "value": 0.000027431500000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000027474124999999997,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/TPU/Default",
            "value": 0.00003825105,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.00092974025,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.00021788887499999998,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAll",
            "value": 0.003107237125,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000953014875,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00418049675,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000027484125,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.000027469624999999997,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003085536,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/XLA",
            "value": 0.0008673156249999999,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/TPU/StructuredTensors",
            "value": 0.00020846315,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/XLA",
            "value": 0.0029596773749999998,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/TPU/Default",
            "value": 0.0002076497,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00418124575,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0041795345,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.003099862375,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.003100285625,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.004180121375,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/TPU/StructuredTensors",
            "value": 0.00003907675,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.00092957575,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.00310690125,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisablePad",
            "value": 0.000006096375,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004180981375,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027526875,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.000027473249999999997,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/TPU/Default",
            "value": 0.0212349164,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.00309297675,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.0041798206249999996,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisablePad",
            "value": 0.000953080125,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/TPU/StructuredTensors",
            "value": 0.00001820395,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.004179822,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001699147625,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAll",
            "value": 0.0000273955,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027461,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.0041799012500000005,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000215666375,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000217882875,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.0030856103749999997,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004181332625,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/XLA",
            "value": 0.000027612875,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027368,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00308517775,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "William Moses",
            "username": "wsmoses",
            "email": "gh@wsmoses.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "f54f7b6955d98dcbda4bed9b1060165d6374ae8b",
          "message": "Change concat_to_pad_comm to 0 and add concat_to_dus (#2033)\n\n* Change concat_to_pad_comm to 0 and add concat_to_dus\n\n* Update Reactant_jll version to 0.0.287",
          "timestamp": "2025-12-26T23:17:26Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/f54f7b6955d98dcbda4bed9b1060165d6374ae8b"
        },
        "date": 1766807823933,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAll",
            "value": 0.005849873,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAll",
            "value": 0.005687029,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005551123,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00646199,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.006196854,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CPU/Default",
            "value": 0.016389087,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.005798433,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.19528075,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CPU/Julia",
            "value": 5.60651612,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.005511059,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CPU/Default",
            "value": 0.059420407,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.005884134,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CPU/Default",
            "value": 0.347008379,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CPU/Default",
            "value": 0.01099439,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001891109,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGatherPad",
            "value": 0.001941942,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.006008093,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CPU/StructuredTensors",
            "value": 0.013678702,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/XLA",
            "value": 0.002307257,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisablePad",
            "value": 0.088621882,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.180763923,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGather",
            "value": 0.001826122,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.005827953,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.193707106,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.184301366,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CPU/Julia",
            "value": 0.020907774,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CPU/Julia",
            "value": 0.006353359,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.005767705,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.006116645,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableTransposeReshape",
            "value": 0.001792004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.199275882,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CPU/StructuredTensors",
            "value": 0.640468734,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.182528948,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisablePad",
            "value": 0.001862782,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CPU/Julia",
            "value": 0.109387591,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.005578572,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGatherPad",
            "value": 0.087027656,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CPU/Julia",
            "value": 0.6169076870000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAll",
            "value": 0.199222026,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CPU/StructuredTensors",
            "value": 3.94004658,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/XLA",
            "value": 0.182508972,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.19434581,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/XLA",
            "value": 0.087644672,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.196076867,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.186163733,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableTransposeReshape",
            "value": 0.089263173,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAll",
            "value": 0.19501177,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.195276206,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.197440668,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CPU/StructuredTensors",
            "value": 0.117545045,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.183708857,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.199333705,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGather",
            "value": 0.088302885,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.00613808,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.087447407,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/XLA",
            "value": 0.006438077,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CPU/StructuredTensors",
            "value": 0.029666581,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00588655,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CPU/Default",
            "value": 2.539375345,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.001097556,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000594489,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001116958,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.003147881,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.002899356,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.00058245,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007120943,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.007113876,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001904325,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003309234,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.00313136,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CUDA/StructuredTensors",
            "value": 0.001477614,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000625708,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003284434,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisablePad",
            "value": 0.001096084,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.001903304,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.003108568,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003130687,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003130612,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000594527,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003280985,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.007112113,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.002835914,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002831231,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/XLA",
            "value": 0.011572972,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000625096,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.007110139,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.0031389,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000231152,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/XLA",
            "value": 0.003319342,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007104474,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001902191,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007182237,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007184585,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisablePad",
            "value": 0.00022992,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001078321,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003176399,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CUDA/Default",
            "value": 0.001923037,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000588341,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.000593145,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007122488,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007115152,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.007116535,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007129381,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/XLA",
            "value": 0.007233615,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/XLA",
            "value": 0.000231002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007201445,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableTransposeReshape",
            "value": 0.000217193,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000585173,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAll",
            "value": 0.000590114,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000593625,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CUDA/StructuredTensors",
            "value": 0.110402093,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CUDA/StructuredTensors",
            "value": 0.002384557,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.0031419,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/XLA",
            "value": 0.001136945,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.001903323,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007183832,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000587331,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.00717957,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/CUDA/Default",
            "value": 0.040219808,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/XLA",
            "value": 0.001935078,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CUDA/StructuredTensors",
            "value": 0.023264005,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.002748121,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/XLA",
            "value": 0.000701749,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000586086,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/CUDA/StructuredTensors",
            "value": 0.006981574,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.003133263,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/XLA",
            "value": 0.003095489,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001899784,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003162365,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.00059761,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/CUDA/Default",
            "value": 0.000572437,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/CUDA/Default",
            "value": 0.008580445,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.00712053,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.011145307,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/CUDA/Default",
            "value": 0.000825316,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001080817,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003138027,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.000597045,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.000578569,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGatherPad",
            "value": 0.000228896,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGather",
            "value": 0.000227459,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.003100563,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.002860231,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00298778,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/TPU/StructuredTensors",
            "value": 0.021074706,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.002988703,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/XLA",
            "value": 0.000586967,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGatherPad",
            "value": 0.000006112,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.000217674,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/XLA",
            "value": 0.000006368,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003093198,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000953048,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.004181253,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.00310016,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableTransposeReshape",
            "value": 0.000006113,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003093113,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.00098514,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/XLA",
            "value": 0.001138376,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000027156,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000928849,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/TPU/Default",
            "value": 0.001523056,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004179601,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGather",
            "value": 0.00000617,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.004180001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000952979,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.002987512,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/XLA",
            "value": 0.004053923,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/XLA",
            "value": 0.002019347,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004181732,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006024,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004181157,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.000027219,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.004180038,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0000272,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.000027238,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000928875,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000217711,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085771,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/TPU/Default",
            "value": 0.000017379,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.000027182,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000929461,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000027164,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[2048 x 2048] primal/TPU/StructuredTensors",
            "value": 0.001539683,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAll",
            "value": 0.000027264,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000027173,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/TPU/Default",
            "value": 0.000038266,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000929453,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000217703,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAll",
            "value": 0.003107183,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000953007,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004180192,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000027218,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.000027216,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003085545,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/XLA",
            "value": 0.000866944,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/TPU/StructuredTensors",
            "value": 0.000208193,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/XLA",
            "value": 0.002959804,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[1024 x 1024] primal/TPU/Default",
            "value": 0.000207407,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004179862,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004179286,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.00309374,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.003121318,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.004180315,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[512 x 512] primal/TPU/StructuredTensors",
            "value": 0.000039097,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929467,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003093533,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisablePad",
            "value": 0.000006103,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004180445,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.00002721,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.000027172,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[4096 x 4096] primal/TPU/Default",
            "value": 0.021097979,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.003100242,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004179322,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisablePad",
            "value": 0.000953081,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz/[256 x 256] primal/TPU/StructuredTensors",
            "value": 0.000018177,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.004180348,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001697171,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAll",
            "value": 0.00002716,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027132,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.004181173,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000215589,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000217853,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.003085572,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004182025,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/XLA",
            "value": 0.000027251,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027162,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.003085281,
            "unit": "s"
          }
        ]
      }
    ]
  }
}