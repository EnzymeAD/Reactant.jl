window.BENCHMARK_DATA = {
  "lastUpdate": 1767907993492,
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
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
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
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/Julia",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.050233903079999996,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
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
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.0020048578000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.0049879029000000005,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CPU/StructuredTensors",
            "value": 0.02605822912,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.091548658,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.162858478,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/Default",
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
            "name": "NewtonSchulz [512 x 512]/primal/CPU/Default",
            "value": 0.013868821239999997,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.008000193479999999,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CPU/Julia",
            "value": 0.019337691,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.16490664840000002,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
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
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.005542915000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGather",
            "value": 0.0759745472,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/StructuredTensors",
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
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
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
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
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
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
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
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
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
            "name": "NewtonSchulz [512 x 512]/primal/CUDA/Default",
            "value": 0.0008220574800000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
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
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
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
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
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
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.0033419316000000004,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CUDA/Default",
            "value": 0.007045649040000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
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
            "name": "NewtonSchulz [512 x 512]/primal/CUDA/StructuredTensors",
            "value": 0.00239302196,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.0015108134,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0005933663000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.0011370460999999999,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CUDA/StructuredTensors",
            "value": 0.02274414568,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.0028274216000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
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
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
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
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
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
            "name": "NewtonSchulz [512 x 512]/primal/TPU/Default",
            "value": 0.000038255049999999996,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
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
            "name": "NewtonSchulz [512 x 512]/primal/TPU/StructuredTensors",
            "value": 0.00003905985,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
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
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
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
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004054636125,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.00020819085000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
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
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/TPU/StructuredTensors",
            "value": 0.0015397231,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/TPU/Default",
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
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
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
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/Default",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/Julia",
            "value": 0.6206285340000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.007260202,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.005997547800000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.06311101572,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.130487122,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
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
            "name": "NewtonSchulz [512 x 512]/primal/CPU/Default",
            "value": 0.01855256352,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.005686961,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
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
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
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
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.19661778439999997,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.21192997620000006,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.08916173289999998,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/StructuredTensors",
            "value": 0.6834012007999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.2119480186,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.45450974136,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.012050517360000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.1958772503,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CPU/Julia",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.12757390500000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.21190348709999998,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CPU/StructuredTensors",
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
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.0060118013,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
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
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/CUDA/StructuredTensors",
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
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
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
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.0032704222,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.0071180865,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CUDA/Default",
            "value": 0.00885404928,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.0018942668000000002,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CUDA/StructuredTensors",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.0019230845600000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007228472100000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
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
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
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
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.0019150535999999999,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.0026594043000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.0007165272,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CUDA/Default",
            "value": 0.00089074872,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
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
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
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
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
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
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
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
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006287875,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003093243625,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.0000173466,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGather",
            "value": 0.0009531035,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/TPU/StructuredTensors",
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
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
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
            "name": "NewtonSchulz [512 x 512]/primal/TPU/Default",
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
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004054005125,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/TPU/Default",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.0208082734,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0030852106250000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.0008670683749999999,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/TPU/StructuredTensors",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.00020748845,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929343625,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
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
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
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
            "name": "NewtonSchulz [512 x 512]/primal/CPU/Default",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 5.369871199,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.004914326,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.05128447744,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.0049539862,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/Default",
            "value": 0.32934130619999996,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
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
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.012156836840000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
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
            "name": "NewtonSchulz [512 x 512]/primal/CPU/Julia",
            "value": 0.019718587000000003,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/StructuredTensors",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/Julia",
            "value": 0.538865346,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAll",
            "value": 0.1842255494,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 3.7749595590800005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.1755046017,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.18792647540000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
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
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.0047172185,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CPU/StructuredTensors",
            "value": 0.027830190520000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0048309333,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
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
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
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
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
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
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
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
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007216101399999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.1106216538,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CUDA/StructuredTensors",
            "value": 0.00238447084,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.0031359261,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.04021056943999999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.0019343765999999998,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CUDA/StructuredTensors",
            "value": 0.0230508122,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.0027398999999999995,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.0007352013999999999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.0006495371,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.007238643320000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0031037981999999997,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
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
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.0005883593199999999,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CUDA/Default",
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
            "name": "NewtonSchulz [512 x 512]/primal/CUDA/Default",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.02115795855,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.002984138125,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
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
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
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
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/TPU/Default",
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
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.00405384875,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
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
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/TPU/StructuredTensors",
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
            "name": "NewtonSchulz [512 x 512]/primal/TPU/Default",
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
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.0008673156249999999,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.00020846315,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.0029596773749999998,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
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
            "name": "NewtonSchulz [512 x 512]/primal/TPU/StructuredTensors",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
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
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
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
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
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
            "name": "NewtonSchulz [512 x 512]/primal/CPU/Default",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 5.60651612,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.005511059,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.059420407,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.005884134,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/Default",
            "value": 0.347008379,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
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
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.013678702,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
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
            "name": "NewtonSchulz [512 x 512]/primal/CPU/Julia",
            "value": 0.020907774,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/StructuredTensors",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/Julia",
            "value": 0.6169076870000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAll",
            "value": 0.199222026,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 3.94004658,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.182508972,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.19434581,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
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
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.006438077,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CPU/StructuredTensors",
            "value": 0.029666581,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00588655,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
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
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
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
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
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
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
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
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
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
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007233615,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.110402093,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CUDA/StructuredTensors",
            "value": 0.002384557,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.0031419,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040219808,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.001935078,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CUDA/StructuredTensors",
            "value": 0.023264005,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.002748121,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.000701749,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000586086,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.006981574,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.003133263,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
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
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.000572437,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CUDA/Default",
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
            "name": "NewtonSchulz [512 x 512]/primal/CUDA/Default",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.021074706,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.002988703,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
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
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
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
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/TPU/Default",
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
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004053923,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
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
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
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
            "name": "NewtonSchulz [2048 x 2048]/primal/TPU/StructuredTensors",
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
            "name": "NewtonSchulz [512 x 512]/primal/TPU/Default",
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
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866944,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000208193,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002959804,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
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
            "name": "NewtonSchulz [512 x 512]/primal/TPU/StructuredTensors",
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
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
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
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
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
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
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
          "id": "7ed9fa4c25e96f597990bb6a0470b851b4a080f4",
          "message": "feat: builtin profiler in the terminal (#2036)\n\n* feat: builtin profiler in the terminal\n\n* test: avoid windows\n\n* Update test/profiling.jl\n\nCo-authored-by: github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>\n\n* docs: add how its printed\n\n---------\n\nCo-authored-by: github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>",
          "timestamp": "2025-12-27T23:53:59Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/7ed9fa4c25e96f597990bb6a0470b851b4a080f4"
        },
        "date": 1766894408540,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAll",
            "value": 0.004878292,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAll",
            "value": 0.005030839,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004818583,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004950102,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.005008489,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CPU/Default",
            "value": 0.014291272,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.005393676,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.186190755,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 5.353680263,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.005045825,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.051880024,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.005177309,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/Default",
            "value": 0.331408198,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.009654512,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001540098,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGatherPad",
            "value": 0.001455534,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004778496,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.012240227,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.001841412,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisablePad",
            "value": 0.081151913,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.171138451,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGather",
            "value": 0.001359949,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004868233,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.183330148,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.170250008,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CPU/Julia",
            "value": 0.020589605,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.0058210020000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.005063185,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.004999287,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableTransposeReshape",
            "value": 0.001326325,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.183568781,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/StructuredTensors",
            "value": 0.609006319,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.172437331,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisablePad",
            "value": 0.001359949,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.09626957000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.004988976,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGatherPad",
            "value": 0.083531961,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/Julia",
            "value": 0.523299091,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAll",
            "value": 0.184941513,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 3.808376465,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.17320179,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.184735044,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.081206842,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.181628571,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.17304014,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableTransposeReshape",
            "value": 0.081329233,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAll",
            "value": 0.186054418,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.181788762,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.186627157,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.108533243,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.172844113,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.183189865,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGather",
            "value": 0.083831628,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.004966386,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.082384644,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.004954962,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CPU/StructuredTensors",
            "value": 0.028044229,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004624824,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.354614525,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.001113973,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000622914,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001121842,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.003183755,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.004213519,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.000603606,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007140289,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.007159669,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.00191221,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003344576,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003151702,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001497024,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000610492,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003297223,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisablePad",
            "value": 0.001083686,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.0019212,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.003129918,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003182914,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003156384,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000623234,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003296626,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.007129623,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.002929007,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002895656,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.011946278,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000635586,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.007127411,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.00312856,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000242546,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003389474,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007140523,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001917802,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007198943,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007197291,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisablePad",
            "value": 0.000248037,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.00110116,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003176998,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001938952,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000613839,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.000616218,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007135731,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007124484,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.007146256,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007151766,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007230444,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000233682,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007217598,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableTransposeReshape",
            "value": 0.000224706,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000601333,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAll",
            "value": 0.000622547,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000594719,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.110627107,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CUDA/StructuredTensors",
            "value": 0.002403627,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.00316598,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.001163646,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.001915934,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007193626,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000601727,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.00719583,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040307234,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.001956721,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CUDA/StructuredTensors",
            "value": 0.022668665,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.002816425,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.00076458,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000624249,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.006851656,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.003156135,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003193082,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001913297,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003153214,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.000610137,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.000609667,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CUDA/Default",
            "value": 0.008288561,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.007133176,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.011914178,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CUDA/Default",
            "value": 0.000841427,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001075378,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003176719,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.000642591,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.000607298,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGatherPad",
            "value": 0.000241435,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGather",
            "value": 0.000244255,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.003337372,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.00294219,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.002987723,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.021049524,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.002985817,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000587774,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGatherPad",
            "value": 0.000006087,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.000217814,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.00000635,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003100312,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000953047,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.004180335,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.003100498,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableTransposeReshape",
            "value": 0.000006153,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003100331,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000985187,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.001138353,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000027408,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000929192,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/TPU/Default",
            "value": 0.001523644,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004179742,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGather",
            "value": 0.00000617,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.004180924,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000952953,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.002985269,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004054729,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002021659,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.00418317,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006114,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004182127,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.000027442,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.004181673,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.000027395,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.000027541,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000929114,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000217794,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003084902,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017387,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.000027504,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000929837,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000027469,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/TPU/StructuredTensors",
            "value": 0.001540351,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAll",
            "value": 0.000027293,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000027452,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/TPU/Default",
            "value": 0.000038247,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.00092974,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000217672,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAll",
            "value": 0.003106967,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000952894,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004180308,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000027387,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.000027431,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003085231,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000867133,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000208446,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002959892,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.000207701,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004181336,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004180049,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.003099685,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.003093153,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.004180408,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/TPU/StructuredTensors",
            "value": 0.000039046,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929739,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.00309991,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisablePad",
            "value": 0.000006123,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004180992,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027443,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.000027497,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.021098604,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.003093286,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004180767,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisablePad",
            "value": 0.000953085,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018245,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.004180804,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001699035,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAll",
            "value": 0.000027387,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027458,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.00418042,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000215775,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000217924,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.003085357,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.00418135,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.0000276,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027476,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.003085321,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]",
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "10d5db4a9d76a9373f44c6d36a92fb54e82b89f6",
          "message": "Update EnzymeAD/Enzyme-JAX to commit 62d30bcf37db1cad5543ceec242f09974e4d0f13 (#2044)\n\nDiff: https://github.com/EnzymeAD/Enzyme-JAX/compare/654e7363760fbcefbafefad8eb4276448da86899...62d30bcf37db1cad5543ceec242f09974e4d0f13\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-12-29T02:45:56Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/10d5db4a9d76a9373f44c6d36a92fb54e82b89f6"
        },
        "date": 1766980797972,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAll",
            "value": 0.005530212,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAll",
            "value": 0.005695768,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005795098,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.005746867,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.005751358,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CPU/Default",
            "value": 0.018519201,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.006184484,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.262337945,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 6.486860444,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.005447593,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.065298087,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.005965643,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/Default",
            "value": 0.401325267,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.00947894,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.002402761,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGatherPad",
            "value": 0.00205376,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.005897324,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.013503183,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.002449501,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisablePad",
            "value": 0.112641423,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.234411027,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGather",
            "value": 0.002166674,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.005477101,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.256226299,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.235433454,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CPU/Julia",
            "value": 0.020968964,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.006878683,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.005856399,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.007074153,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableTransposeReshape",
            "value": 0.00209717,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.258270325,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/StructuredTensors",
            "value": 0.929453871,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.237526358,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisablePad",
            "value": 0.002001658,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.105550268,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.005686559,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGatherPad",
            "value": 0.11212442,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CPU/Julia",
            "value": 0.736342566,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAll",
            "value": 0.254596036,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 5.061920924,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.234521409,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.252858038,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.118103908,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.255707461,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.243089487,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableTransposeReshape",
            "value": 0.111255624,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAll",
            "value": 0.256078295,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.253378214,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.256879539,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.139981248,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.239043943,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.253475359,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGather",
            "value": 0.111143216,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.005528101,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.115782908,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.006068064,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CPU/StructuredTensors",
            "value": 0.031080869,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005970027,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.966312991,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.00108904,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000633519,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001099364,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.003142926,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.002898816,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.000617945,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007121029,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.007128402,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001905302,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0033334,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003161222,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001483474,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000621619,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003292423,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisablePad",
            "value": 0.001111753,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.00190769,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.003117096,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003143642,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003121109,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000635091,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003285361,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.007118402,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.002892438,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002912064,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.011325676,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000634886,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.007118792,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.003119211,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000250508,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003324778,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007119806,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001906951,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007192836,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007187248,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisablePad",
            "value": 0.00023438,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001088589,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003159104,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001918361,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000646758,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.000634109,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007127264,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007115071,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.007100263,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007122631,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007225879,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000238082,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007195405,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableTransposeReshape",
            "value": 0.00022012,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000626361,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAll",
            "value": 0.000630476,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000634623,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.110305362,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CUDA/StructuredTensors",
            "value": 0.002395591,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.003154054,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.001127661,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.001906269,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007194775,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000636016,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007186887,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040187365,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.00193303,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CUDA/StructuredTensors",
            "value": 0.023566819,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.002827923,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.00078334,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000640269,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.007084395,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.003125251,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003168808,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001905473,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003136461,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.000617753,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.00074928,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/CUDA/Default",
            "value": 0.008841319,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.007121995,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.011220252,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/CUDA/Default",
            "value": 0.000833693,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001094899,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003150192,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.00064397,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.000619183,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGatherPad",
            "value": 0.000235637,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGather",
            "value": 0.000233948,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.003130574,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.002903088,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.002986879,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.020889054,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.002984881,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000586621,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGatherPad",
            "value": 0.00000605,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.000217757,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006245,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003093032,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000952996,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.00418202,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.003093337,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableTransposeReshape",
            "value": 0.000006066,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003093495,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000985018,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.001138467,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00002717,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000928963,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/TPU/Default",
            "value": 0.001522887,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004179589,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGather",
            "value": 0.00000608,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.004180697,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000953017,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.002986323,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004053669,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002018942,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004181641,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006107,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004181489,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.000027117,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.004179061,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.000027122,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.000027131,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000928814,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000217694,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.0030851,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017256,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.00002715,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000929311,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000027142,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [2048 x 2048]/primal/TPU/StructuredTensors",
            "value": 0.001539486,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAll",
            "value": 0.000027212,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000027169,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/TPU/Default",
            "value": 0.000038152,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000929328,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000217569,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAll",
            "value": 0.003093366,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000952984,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004180655,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000027181,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.000027178,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003085268,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866808,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000207973,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002959888,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.000207258,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004181512,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004180852,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.003093775,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.003093311,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.004181205,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [512 x 512]/primal/TPU/StructuredTensors",
            "value": 0.000038976,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.00092955,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003093659,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisablePad",
            "value": 0.000005995,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004180047,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027202,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.000027206,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.020895326,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.003093358,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004179822,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisablePad",
            "value": 0.000953008,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018085,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.004180369,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001696828,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAll",
            "value": 0.000027231,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027229,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.004180712,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000215371,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000217574,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.003085029,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004180353,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.000027203,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.00002718,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.003085024,
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
          "id": "715202e239d41dc880f8d385f0915f3e54c235f7",
          "message": "perf: add polybench 4.2 raising benchmarks (#2038)\n\n* perf: polybench with raising\n\n* fix: disable to conv even on TPUs\n\n* chore: bump jll",
          "timestamp": "2025-12-29T18:43:10Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/715202e239d41dc880f8d385f0915f3e54c235f7"
        },
        "date": 1767037113998,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAll",
            "value": 0.005184866,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAll",
            "value": 0.004864704,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004718943,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.627314762,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.061623947,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00477245,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.000804454,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004863659,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.000551609,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default",
            "value": 0.021452967,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Julia",
            "value": 38.318957392,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.023524416,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default_manual_vectorized",
            "value": 0.075707562,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.004838616,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default",
            "value": 0.029170443,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default",
            "value": 0.511748018,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.179914251,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Julia",
            "value": 0.063852371,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 4.962773817,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.004830294,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.006639075,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.020714375,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.050727081,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Julia",
            "value": 0.029976504,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.009570909,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.005030534,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default",
            "value": 0.082822706,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Julia",
            "value": 23.042298052000003,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.009090329,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001407948,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGatherPad",
            "value": 0.001410966,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default",
            "value": 0.000559219,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004942697,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.027048276,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default",
            "value": 0.006795059,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.011371061,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.001413913,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisablePad",
            "value": 0.073306229,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default",
            "value": 0.706932912,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.166662217,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default_manual_vectorized",
            "value": 0.705445627,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.016544926,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGather",
            "value": 0.001382478,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.005170353,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.179662168,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.167065633,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Julia",
            "value": 65.24205360100001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.005772575,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.008054891,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default",
            "value": 0.00335535,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Julia",
            "value": 0.040916225,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.004596794,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.004982979,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Julia",
            "value": 0.365506865,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default",
            "value": 0.006683427,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Julia",
            "value": 1.7926919710000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableTransposeReshape",
            "value": 0.001297449,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.179665538,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.168007261,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisablePad",
            "value": 0.001438851,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.09543939200000001,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Julia",
            "value": 27.469438955,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.004702389,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGatherPad",
            "value": 0.073404647,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAll",
            "value": 0.180147211,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default",
            "value": 0.029730349,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 3.743266054,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Julia",
            "value": 0.0005604550000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.166514557,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.180224097,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Julia",
            "value": 0.20691120500000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.074461926,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default",
            "value": 0.000884919,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Julia",
            "value": 387.22367031100003,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default",
            "value": 0.002273207,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.178204988,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default",
            "value": 0.645645884,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Julia",
            "value": 273.854598421,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.168675022,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default",
            "value": 2.467808596,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableTransposeReshape",
            "value": 0.074714259,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default_manual_vectorized",
            "value": 1.197337806,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.177585509,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAll",
            "value": 0.179507417,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Julia",
            "value": 23.023274894,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.019651261,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.178320054,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.106708294,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.167545915,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Julia",
            "value": 15.313825634,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.179660884,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Julia",
            "value": 45.723757878,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Julia",
            "value": 15.020573444,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default",
            "value": 0.008630521,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGather",
            "value": 0.074607945,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.004989756,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.075604403,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.019324065,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.004710065,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default",
            "value": 0.017443565,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.006944012,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default",
            "value": 3.890974354,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004464578,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.322697147,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.001080059,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000589662,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001118354,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.003154025,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000057488,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.002910659,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007119983,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.007118123,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001906201,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.000596206,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003309013,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default",
            "value": 0.037024575,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003137235,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001482742,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000025805,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000206878,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000578887,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.028206881,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003300103,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisablePad",
            "value": 0.001075464,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.001906697,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default",
            "value": 0.000446993,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.00313086,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003154564,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003128034,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000590727,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000156641,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003270667,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.007106156,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.002847968,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.020220737,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002865521,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.011500746,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000597125,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.007094126,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000442327,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.003108228,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000231764,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003328967,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007126805,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.00026274,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default",
            "value": 0.061886277,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default",
            "value": 0.000162375,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default_manual_vectorized",
            "value": 0.001473317,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.024181646,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001906826,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007177,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007175214,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default",
            "value": 0.000287892,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisablePad",
            "value": 0.000228241,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001071659,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003160231,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001928307,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000582871,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.000591522,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default",
            "value": 0.013194216,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000111226,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007111321,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007117275,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.007101507,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007227068,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00712292,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default",
            "value": 0.002660822,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000233818,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007185269,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableTransposeReshape",
            "value": 0.000213322,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000581848,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAll",
            "value": 0.000585281,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000584811,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.110823401,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.003141351,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000255098,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default",
            "value": 0.000045341,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.001151611,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.00190942,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007165801,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000576706,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default",
            "value": 0.000055943,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000056017,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.00717235,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default",
            "value": 0.00046355,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default",
            "value": 0.020406156,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.013188262,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040361514,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.001933298,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default",
            "value": 0.02028123,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000259508,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.002738679,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.000720213,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000594983,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000466948,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.007181194,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.003123893,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003081936,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001906274,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003150379,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.000579332,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.000579156,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default",
            "value": 0.000027383,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default",
            "value": 0.000106292,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.007117317,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000499268,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.010980393,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001110459,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003138186,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.00059178,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.000584084,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGatherPad",
            "value": 0.000231255,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGather",
            "value": 0.000228887,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default",
            "value": 0.000110551,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.003126256,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default",
            "value": 0.0002565,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default",
            "value": 0.023848261,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.002803956,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.002987042,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.021308444,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.00298427,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.00058644,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGatherPad",
            "value": 0.000006117,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.000217803,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006287,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003093398,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000953039,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.004180044,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.003093463,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableTransposeReshape",
            "value": 0.000006099,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003093329,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000031664,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default",
            "value": 0.000072517,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000985085,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.001138618,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000016497,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000024113,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000929343,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000027524,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004179663,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGather",
            "value": 0.000006144,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default",
            "value": 0.04305566,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000075093,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.004180932,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000953105,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004052921,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00298904,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002021642,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004181667,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006072,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004182202,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.000027453,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default_manual_vectorized",
            "value": 0.001087799,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.004180459,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.000027354,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.000027539,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000929115,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default",
            "value": 0.001650772,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000217979,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.00308514,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default",
            "value": 0.000045009,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017385,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.000027488,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default",
            "value": 0.000035647,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000929694,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000058406,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default",
            "value": 0.019597231,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000027486,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000047725,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default",
            "value": 0.261753582,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAll",
            "value": 0.000027584,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000027425,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000037562,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000929681,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000023432,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000217909,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAll",
            "value": 0.003100136,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000953102,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default",
            "value": 0.000016465,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default",
            "value": 0.009866725,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004179829,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000027518,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default",
            "value": 0.000030385,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000053236,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.000027589,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003085459,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866785,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.00020833,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002960073,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.00020775,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004181535,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004181248,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.003100553,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.00309298,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.004180432,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929545,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.005708148,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003100684,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisablePad",
            "value": 0.000006041,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004180561,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027421,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.000027549,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.02134776,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.003093133,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004180923,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default",
            "value": 0.000087374,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default",
            "value": 0.000051823,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000087436,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisablePad",
            "value": 0.00095304,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018207,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default",
            "value": 0.00002344,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.027860303,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.022079334,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default",
            "value": 0.026676122,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000072696,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.004181166,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001698882,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default",
            "value": 0.000086627,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAll",
            "value": 0.000027531,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027559,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.004180856,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000215869,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default",
            "value": 0.000024163,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000218062,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.003085675,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004180474,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.000027585,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027563,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000045499,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.003085775,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.261753598,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default",
            "value": 0.001651527,
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
          "id": "715202e239d41dc880f8d385f0915f3e54c235f7",
          "message": "perf: add polybench 4.2 raising benchmarks (#2038)\n\n* perf: polybench with raising\n\n* fix: disable to conv even on TPUs\n\n* chore: bump jll",
          "timestamp": "2025-12-29T18:43:10Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/715202e239d41dc880f8d385f0915f3e54c235f7"
        },
        "date": 1767068866741,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAll",
            "value": 0.005460981,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAll",
            "value": 0.005488969,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005109849,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.675107566,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.06245174,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.005103033,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.000973964,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00533378,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.000460227,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default",
            "value": 0.021762681,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Julia",
            "value": 41.290547456000006,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.02079031,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default_manual_vectorized",
            "value": 0.078010584,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.00522963,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default",
            "value": 0.02975665,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default",
            "value": 0.549050186,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.20111952,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Julia",
            "value": 0.06322050400000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 4.983853728000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.00497954,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.007758554,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.023544645,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.056919679,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Julia",
            "value": 0.029548078000000002,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.010228619,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.005283509,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default",
            "value": 0.088288417,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Julia",
            "value": 23.221170958000002,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.010916773,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001547547,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGatherPad",
            "value": 0.001439245,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default",
            "value": 0.00055517,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.005288909,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.027719046,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default",
            "value": 0.007315012,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.012432726,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.001469136,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisablePad",
            "value": 0.080032523,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default",
            "value": 0.855571799,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.186026441,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default_manual_vectorized",
            "value": 0.84967254,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.016669583,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGather",
            "value": 0.001512704,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.005118077,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.202387277,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.180731074,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Julia",
            "value": 67.92936782800001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.006011181,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.008649695,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default",
            "value": 0.003985624,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Julia",
            "value": 0.043840496,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.005359205,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.005044817,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Julia",
            "value": 0.379032655,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default",
            "value": 0.006315068,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Julia",
            "value": 1.676401422,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableTransposeReshape",
            "value": 0.001320284,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.193141034,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.183873575,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisablePad",
            "value": 0.001509058,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.10253864,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Julia",
            "value": 26.669753712000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.005176702,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGatherPad",
            "value": 0.081510458,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAll",
            "value": 0.204033285,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default",
            "value": 0.032538609,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 3.81934792,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Julia",
            "value": 0.000562898,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.180755603,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.20350813,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Julia",
            "value": 0.20605087800000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.08096341,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default",
            "value": 0.000916646,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Julia",
            "value": 416.03841231900003,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default",
            "value": 0.002791641,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.196227629,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default",
            "value": 0.716226576,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Julia",
            "value": 289.49439094800005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.188078645,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default",
            "value": 3.231181822,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableTransposeReshape",
            "value": 0.080309077,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default_manual_vectorized",
            "value": 1.412408906,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.201439547,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAll",
            "value": 0.202031459,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Julia",
            "value": 23.131970043000003,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.020485855,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.201705743,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.117259784,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.187958108,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Julia",
            "value": 15.350262391000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.194575791,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Julia",
            "value": 50.201675246,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Julia",
            "value": 14.037770316000001,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default",
            "value": 0.008574064,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGather",
            "value": 0.081659904,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.005483348,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.081190824,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.023892954,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.005142453,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default",
            "value": 0.020829447,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.007820695,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default",
            "value": 4.173858159,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004958486,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.394766892,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.00107599,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.00062706,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001111375,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.003136073,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000058966,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.002861358,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007133002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.007135936,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001910162,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.000616916,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003321422,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default",
            "value": 0.036157151,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003128334,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001483382,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000025856,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.00020612,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000611216,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.02709133,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003280249,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisablePad",
            "value": 0.001056623,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.001914926,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default",
            "value": 0.000453257,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.003106181,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003122904,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003133438,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.00061241,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000158027,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003275667,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.007122576,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.002845561,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.020448606,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.00284004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.011480233,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000628856,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.00711939,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000445919,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.003104936,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000235723,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003307321,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007131899,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.00026601,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default",
            "value": 0.063284273,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default",
            "value": 0.000161397,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default_manual_vectorized",
            "value": 0.001485311,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.024637869,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001914372,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007196639,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007192374,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default",
            "value": 0.000290983,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisablePad",
            "value": 0.00023611,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001064829,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003147539,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001937681,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000619578,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.000624855,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default",
            "value": 0.013247635,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000112095,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007120511,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007138333,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.007121164,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.00724465,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007133405,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default",
            "value": 0.002262888,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000229733,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007208933,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableTransposeReshape",
            "value": 0.000220782,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000605454,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAll",
            "value": 0.000619857,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000610618,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.110737404,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.003139601,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000256724,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default",
            "value": 0.000043123,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.001134824,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.001915828,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007190585,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000607204,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default",
            "value": 0.000054745,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000054621,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007199596,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default",
            "value": 0.000467282,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default",
            "value": 0.020750414,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.013248737,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040359216,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.001937681,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default",
            "value": 0.020423413,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000265646,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.002762951,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.000749825,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000619959,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000470259,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.006892052,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.003133336,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003130835,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001917848,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003136686,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.000621255,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.000735267,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default",
            "value": 0.000025971,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default",
            "value": 0.000106303,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.007132307,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000502767,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.011285993,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001064489,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00316203,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.000614519,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.000609614,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGatherPad",
            "value": 0.000234293,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGather",
            "value": 0.000234254,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default",
            "value": 0.000111973,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.003141709,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default",
            "value": 0.000258088,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default",
            "value": 0.025177274,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.002840889,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.002985876,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.021646862,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.002988943,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000586512,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGatherPad",
            "value": 0.000006064,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.00021795,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006294,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00309336,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGather",
            "value": 0.00095307,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.00418104,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.003100032,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableTransposeReshape",
            "value": 0.000006063,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003093077,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000031429,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default",
            "value": 0.000072604,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000985205,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.001138371,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000016397,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000024126,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000928911,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00002728,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004178744,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGather",
            "value": 0.000006123,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default",
            "value": 0.043055591,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000075103,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.004180132,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000953121,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.00405307,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.002988115,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002020025,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004180884,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006105,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004179957,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.00002732,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default_manual_vectorized",
            "value": 0.00111537,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.004179285,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00002732,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.000027313,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000929105,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default",
            "value": 0.001650577,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000217931,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085236,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default",
            "value": 0.000045101,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017407,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.000027261,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default",
            "value": 0.000035669,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000929452,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000058182,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default",
            "value": 0.019597322,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000027333,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000047657,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default",
            "value": 0.261753576,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAll",
            "value": 0.000027295,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000027336,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.00003746,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000929427,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000023489,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000217914,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAll",
            "value": 0.003100215,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000953114,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default",
            "value": 0.000016355,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default",
            "value": 0.00998273,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004180079,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00002718,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default",
            "value": 0.000030357,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000053281,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.000027275,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003085083,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866692,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000208182,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.00295924,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.00020755,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004181594,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004179278,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.003093485,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.003093778,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.004180913,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929559,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.005707797,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003093191,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisablePad",
            "value": 0.000006122,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004179464,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027352,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.000027327,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.021696778,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.003093045,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004180371,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default",
            "value": 0.000087362,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default",
            "value": 0.000051682,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000087441,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisablePad",
            "value": 0.000953168,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018229,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default",
            "value": 0.0000234,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.027859918,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.022079372,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default",
            "value": 0.02667625,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000072603,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.004180517,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001697355,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default",
            "value": 0.000086489,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAll",
            "value": 0.000027284,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027237,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.004179828,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000215755,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default",
            "value": 0.000024196,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000217899,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.00308555,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004180352,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.000027364,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027233,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000045495,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.003085616,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.261753555,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default",
            "value": 0.001650449,
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
          "id": "715202e239d41dc880f8d385f0915f3e54c235f7",
          "message": "perf: add polybench 4.2 raising benchmarks (#2038)\n\n* perf: polybench with raising\n\n* fix: disable to conv even on TPUs\n\n* chore: bump jll",
          "timestamp": "2025-12-29T18:43:10Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/715202e239d41dc880f8d385f0915f3e54c235f7"
        },
        "date": 1767155067278,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAll",
            "value": 0.005474905,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAll",
            "value": 0.005794634,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.006018102,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.712775882,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.069486025,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.005548278,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.001125648,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.006919471,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.000456194,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default",
            "value": 0.025891471,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Julia",
            "value": 41.381931135,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.02114716,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default_manual_vectorized",
            "value": 0.082660028,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.005367957,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default",
            "value": 0.029970957,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default",
            "value": 0.546491798,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.200065758,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Julia",
            "value": 0.06325541400000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 5.270853895,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.005412317,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.006841586,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.024069278,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.05529859,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Julia",
            "value": 0.030839599000000002,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.010392968,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.005493687,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default",
            "value": 0.091687912,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Julia",
            "value": 23.196986509000002,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.010888399,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001773262,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGatherPad",
            "value": 0.001778669,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default",
            "value": 0.000560373,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00522042,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.027518362,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default",
            "value": 0.007530648,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.013635935,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.001745724,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisablePad",
            "value": 0.081440122,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default",
            "value": 0.969568373,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.185673284,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default_manual_vectorized",
            "value": 0.966701077,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.016502752,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGather",
            "value": 0.001561438,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.005423081,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.198354969,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.186174746,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Julia",
            "value": 70.096223765,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.0059927190000000005,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.008841661,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default",
            "value": 0.003784566,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Julia",
            "value": 0.040412251,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.005628115,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.005842241,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Julia",
            "value": 0.34924152,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default",
            "value": 0.006961406,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Julia",
            "value": 1.597039598,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableTransposeReshape",
            "value": 0.001412481,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.199561104,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.181772147,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisablePad",
            "value": 0.00163966,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.09821020300000001,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Julia",
            "value": 27.717604462,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.005612504,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGatherPad",
            "value": 0.083357967,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAll",
            "value": 0.197265499,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default",
            "value": 0.034361036,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 3.90162044,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Julia",
            "value": 0.000575272,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.188795154,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.197131936,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Julia",
            "value": 0.208458734,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.084362658,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default",
            "value": 0.000906514,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Julia",
            "value": 272.05363400600004,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default",
            "value": 0.002774429,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.201553329,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default",
            "value": 0.764992015,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Julia",
            "value": 297.178149281,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.189285746,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default",
            "value": 3.330496028,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableTransposeReshape",
            "value": 0.084957988,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default_manual_vectorized",
            "value": 1.671060773,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.20633653,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAll",
            "value": 0.201572288,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Julia",
            "value": 23.196644798,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.019565139,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.200392842,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.118051612,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.189127811,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Julia",
            "value": 15.447666129000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.200341574,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Julia",
            "value": 52.366579715,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Julia",
            "value": 14.408057931,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default",
            "value": 0.008978814,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGather",
            "value": 0.081862954,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.005445089,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.083272385,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.022943192,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.005552266,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default",
            "value": 0.020664707,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.007519219,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default",
            "value": 4.350347475,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005644042,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.437587512,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.001053774,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000616,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.00108307,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.003108229,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000057517,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.00281191,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007082083,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.007081809,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001894066,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.000653089,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003273908,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default",
            "value": 0.0361631,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003083898,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001460058,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000025268,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000206125,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000627268,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.027704906,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003249639,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisablePad",
            "value": 0.00104353,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.001894766,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default",
            "value": 0.000444324,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.003075486,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003113405,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003098894,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000608785,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000158935,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003239245,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.007073676,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.002812332,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.020709288,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002822955,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.01131204,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000620983,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.007071971,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000440004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.003069348,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000230762,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003261032,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007079164,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000257943,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default",
            "value": 0.06305962,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default",
            "value": 0.000157217,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default_manual_vectorized",
            "value": 0.001879985,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.024445511,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001893941,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007149511,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007154442,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default",
            "value": 0.000278391,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisablePad",
            "value": 0.000236141,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001048613,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003103518,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001907988,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000625204,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.000618734,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default",
            "value": 0.012480074,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000148839,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007069469,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007087986,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.007076915,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007195353,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007082218,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default",
            "value": 0.00224357,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000229729,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007170365,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableTransposeReshape",
            "value": 0.000213869,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000616522,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAll",
            "value": 0.00062073,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000588052,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.110468393,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.003103428,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000254343,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default",
            "value": 0.000043524,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.001403157,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.001894202,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007149362,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000602337,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default",
            "value": 0.000054967,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000054881,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007151384,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default",
            "value": 0.000461233,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default",
            "value": 0.020529187,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.013008471,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040204894,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.001915531,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default",
            "value": 0.020622685,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000259152,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.002738632,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.000714686,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000611575,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.00046504,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.007186916,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.003094446,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003058811,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001892471,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003091292,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.000596861,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.00058775,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default",
            "value": 0.000025655,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default",
            "value": 0.000106164,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.00708331,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000497546,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.011095106,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001046375,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00310756,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.000614794,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.0006001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGatherPad",
            "value": 0.000231207,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGather",
            "value": 0.000233149,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default",
            "value": 0.000112401,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.00306962,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default",
            "value": 0.000255796,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default",
            "value": 0.02481965,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.002787061,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00298316,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.020837467,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.002983534,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000586056,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGatherPad",
            "value": 0.000006055,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.00021774,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006177,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003093359,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000952989,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.004180928,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.003092935,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableTransposeReshape",
            "value": 0.000006051,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003093095,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.00003152,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default",
            "value": 0.000072556,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000985175,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.001138582,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000016321,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000024156,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000928868,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00002725,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004178601,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGather",
            "value": 0.000006024,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default",
            "value": 0.043055602,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000074949,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.004179979,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000952959,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.0040519,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.002988837,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002020402,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004181059,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006069,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004181132,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.00002719,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default_manual_vectorized",
            "value": 0.001086367,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.004179235,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.000027169,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.000027255,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000928773,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default",
            "value": 0.001574987,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.00021778,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085373,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default",
            "value": 0.000045067,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017351,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.000027305,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default",
            "value": 0.000035648,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000929292,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000058096,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default",
            "value": 0.019597132,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000027206,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000047588,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default",
            "value": 0.261753673,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAll",
            "value": 0.000027228,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000027197,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.00003744,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000929433,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000023449,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000217688,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAll",
            "value": 0.003092703,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000952938,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default",
            "value": 0.000016357,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default",
            "value": 0.009983146,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004180371,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000027144,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default",
            "value": 0.000030393,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000053228,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.000027235,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003084931,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866645,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.0002081,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002960056,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.000207452,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004181179,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004180234,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.003093192,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.003093761,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.004179177,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929281,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.00570805,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003093466,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisablePad",
            "value": 0.000006012,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004180064,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027243,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.000027221,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.021004633,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.003093241,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004178732,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default",
            "value": 0.000087494,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default",
            "value": 0.00005158,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000087488,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisablePad",
            "value": 0.000953065,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018154,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default",
            "value": 0.000023398,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.027860272,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.02207938,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default",
            "value": 0.026676207,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000072709,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.004179887,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001697403,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default",
            "value": 0.000086465,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAll",
            "value": 0.000027187,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027212,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.004180351,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000215376,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default",
            "value": 0.000024149,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000217834,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.003085216,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004180729,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.000027375,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027225,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000045452,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.003085321,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.261753693,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default",
            "value": 0.001650438,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]",
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "a90ac9e53e6ade2a902d18437999128816b195eb",
          "message": "Update EnzymeAD/Enzyme-JAX to commit 426a717b0f6c246c205f72ebeaf67727b028acf7 (#2045)\n\nDiff: https://github.com/EnzymeAD/Enzyme-JAX/compare/62d30bcf37db1cad5543ceec242f09974e4d0f13...426a717b0f6c246c205f72ebeaf67727b028acf7\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-12-31T18:21:55Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/a90ac9e53e6ade2a902d18437999128816b195eb"
        },
        "date": 1767241446842,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAll",
            "value": 0.004598819,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAll",
            "value": 0.004616704,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004445045,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.591358295,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.061585228,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004661701,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.000766104,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004804625,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.000470537,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default",
            "value": 0.022851433,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Julia",
            "value": 40.708006145000006,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.025335363,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default_manual_vectorized",
            "value": 0.074579507,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.004529679,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default",
            "value": 0.027402588,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default",
            "value": 0.503176218,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.172137999,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Julia",
            "value": 0.062960953,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 4.847843357,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.004788194,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.006844119,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.026985811,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.049829318,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Julia",
            "value": 0.026099960000000002,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.00996555,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAll",
            "value": 0.004520407,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default",
            "value": 0.08131621,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Julia",
            "value": 23.058697247,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.008754732,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001425266,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGatherPad",
            "value": 0.001412241,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default",
            "value": 0.000539804,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00472406,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.027949986,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default",
            "value": 0.006907346,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.011625962,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.001349618,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisablePad",
            "value": 0.070144972,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default",
            "value": 0.768435705,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.161141763,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default_manual_vectorized",
            "value": 0.722337993,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.019194205,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableScatterGather",
            "value": 0.00135214,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004732137,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAfterEnzyme",
            "value": 0.172603441,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.161339835,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Julia",
            "value": 59.61254314600001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.005683657,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.007033136,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default",
            "value": 0.003419554,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Julia",
            "value": 0.039629573,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.004922055,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.004639236,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Julia",
            "value": 0.37398066,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default",
            "value": 0.00679322,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Julia",
            "value": 1.6472644820000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisableTransposeReshape",
            "value": 0.001282429,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.171710062,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.160907487,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/DisablePad",
            "value": 0.001370757,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.093327773,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Julia",
            "value": 26.610403655000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.004524749,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGatherPad",
            "value": 0.07111548,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAll",
            "value": 0.172765459,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default",
            "value": 0.031702765,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 3.684968643,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Julia",
            "value": 0.0005612260000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.160276228,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.172626782,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Julia",
            "value": 0.20460368,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.071871584,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default",
            "value": 0.000892215,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Julia",
            "value": 311.50076511400005,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default",
            "value": 0.002413413,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.172809262,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default",
            "value": 0.655226517,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Julia",
            "value": 290.37794609400004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.160372154,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default",
            "value": 2.18471039,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableTransposeReshape",
            "value": 0.071327739,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default_manual_vectorized",
            "value": 1.108955663,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.171062892,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadAll",
            "value": 0.173168442,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Julia",
            "value": 23.032102311000003,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.020056064,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.172185686,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.102386274,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisablePadBeforeEnzyme",
            "value": 0.161488038,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Julia",
            "value": 15.262810753,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.173954892,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Julia",
            "value": 47.49384098,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Julia",
            "value": 7.272855914000001,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default",
            "value": 0.007197346,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/DisableScatterGather",
            "value": 0.070024082,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.004679511,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.069773192,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.021681959,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.00453908,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default",
            "value": 0.020439494,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.006788362,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default",
            "value": 3.984625455,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004566394,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.285414178,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.00109438,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000611673,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001118142,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.003164818,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000058611,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.002946949,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007127526,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.007122861,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001911257,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.000617704,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003333872,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default",
            "value": 0.035398188,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003147552,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001493139,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000025776,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000206165,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006078,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.027670875,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003291317,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisablePad",
            "value": 0.001082822,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.001915904,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default",
            "value": 0.000448192,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.003126803,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003144051,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003170445,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000611797,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000158742,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003302818,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.007101888,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.002907512,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.020513618,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002899871,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.011264547,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000613637,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.007119912,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000444356,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.003125807,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000237695,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003353416,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007131776,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000261807,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default",
            "value": 0.064337456,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default",
            "value": 0.000159516,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default_manual_vectorized",
            "value": 0.001878051,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.024572379,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001931798,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007191263,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007187825,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default",
            "value": 0.000285473,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisablePad",
            "value": 0.000233205,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGather",
            "value": 0.001075817,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003177841,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001939234,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000626053,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.000616374,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default",
            "value": 0.013203445,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000111539,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007119465,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007127903,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.007120065,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007234308,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00713078,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default",
            "value": 0.001938223,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000235045,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007198699,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableTransposeReshape",
            "value": 0.000220962,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0006063,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisablePadAll",
            "value": 0.000614582,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000588111,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.110589941,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadAfterEnzyme",
            "value": 0.003163094,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000254955,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default",
            "value": 0.000044109,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.001165291,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisablePad",
            "value": 0.001910079,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007191475,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000611107,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default",
            "value": 0.00005431,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000053837,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007193311,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default",
            "value": 0.000463197,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default",
            "value": 0.020635249,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.013367319,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040294831,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.001936178,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default",
            "value": 0.020310532,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000264015,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableTransposeReshape",
            "value": 0.002811096,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.000719486,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.00062045,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000466288,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.007224912,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisablePadBeforeEnzyme",
            "value": 0.003153333,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003240755,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001910147,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003169401,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.000602095,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.000594669,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default",
            "value": 0.000026643,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default",
            "value": 0.000106752,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.007128851,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000499446,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.011514494,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.001086978,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003168393,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.000606271,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.000609401,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGatherPad",
            "value": 0.000236338,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/DisableScatterGather",
            "value": 0.00023861,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default",
            "value": 0.00011191,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.003123711,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default",
            "value": 0.000256396,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default",
            "value": 0.025679239,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/DisableScatterGatherPad",
            "value": 0.002834299,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.002983534,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.020937835,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.002984452,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000586925,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGatherPad",
            "value": 0.00000606,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.000217969,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006307,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003093718,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000953014,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.004180584,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.003093512,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableTransposeReshape",
            "value": 0.000006154,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003093421,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000031572,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default",
            "value": 0.000072592,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000985238,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.001138479,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000016433,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000024181,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000929227,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000027241,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004178655,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisableScatterGather",
            "value": 0.000006106,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default",
            "value": 0.043055548,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.00007506,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.004181294,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000953127,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004052719,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.002989102,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002021105,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004180297,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006033,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00418115,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.000027229,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default_manual_vectorized",
            "value": 0.001086345,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.004178691,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.000027203,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.000027199,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000929179,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default",
            "value": 0.001580417,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000217877,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085494,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default",
            "value": 0.000045049,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017408,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.000027158,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default",
            "value": 0.000035608,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGather",
            "value": 0.000929662,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000058263,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default",
            "value": 0.019597193,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000027237,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000047589,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default",
            "value": 0.261753648,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAll",
            "value": 0.000027224,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000027255,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000037359,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000929648,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000023416,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000217899,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAll",
            "value": 0.003100167,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisableScatterGatherPad",
            "value": 0.000953026,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default",
            "value": 0.000016345,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default",
            "value": 0.00986611,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004179433,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000027237,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default",
            "value": 0.000030353,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000053234,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.000027203,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003085781,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000867003,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000208284,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002960082,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.000207644,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004182291,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004180739,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.003093408,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.003100233,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.00418128,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929469,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.005707927,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003093111,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/DisablePad",
            "value": 0.000006051,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004180545,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027287,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.000027195,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.021061029,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.003093063,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004181469,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default",
            "value": 0.000087433,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default",
            "value": 0.000051736,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000087508,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/DisablePad",
            "value": 0.000953136,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018207,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default",
            "value": 0.000023425,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.02786002,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.022079407,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default",
            "value": 0.026676297,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.00007284,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.004180244,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001698614,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default",
            "value": 0.000086586,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DisableScatterGatherAll",
            "value": 0.000027154,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027154,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DisablePadBeforeEnzyme",
            "value": 0.004180251,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisableTransposeReshape",
            "value": 0.000215677,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default",
            "value": 0.000024086,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/DisablePad",
            "value": 0.000218099,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisablePadAfterEnzyme",
            "value": 0.00308563,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004179749,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.000027358,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027239,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000045539,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.003085413,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.261753636,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default",
            "value": 0.001650668,
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
          "id": "35f3bbcde272a9ed958a1074c1f6e349a080a5fc",
          "message": "chore: bump jll (#2052)\n\n* chore: bump jll\n\n* perf: multiple fixes",
          "timestamp": "2026-01-01T19:14:48Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/35f3bbcde272a9ed958a1074c1f6e349a080a5fc"
        },
        "date": 1767298415178,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.842410411,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.444722149,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.070664708,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.000826416,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.131848252,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.000524588,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default",
            "value": 0.024791279,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.012131832,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.063684441,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Julia",
            "value": 37.796858559,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.026989422,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default_manual_vectorized",
            "value": 0.078312726,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.006083154,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 4.037179695,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default",
            "value": 0.029049106,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default",
            "value": 0.116832229,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Julia",
            "value": 0.066012334,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.014051653,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.006928831,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.026790423,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Julia",
            "value": 0.028982786000000003,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.010352628,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default",
            "value": 0.040670062,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Julia",
            "value": 23.077563411,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001879862,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default",
            "value": 0.000503799,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.041823062,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default",
            "value": 0.00747259,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.001876346,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default",
            "value": 1.011588219,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default_manual_vectorized",
            "value": 0.950464458,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.018073822,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.193966866,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Julia",
            "value": 83.76527566300001,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.007943564,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default",
            "value": 0.003916943,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Julia",
            "value": 0.040608243,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.006022848,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Julia",
            "value": 0.36457182000000005,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default",
            "value": 0.006376141,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Julia",
            "value": 1.682883327,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.12717868200000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.20803854,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Julia",
            "value": 26.426327286000003,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default",
            "value": 0.030227729,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Julia",
            "value": 0.000568466,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.194426183,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Julia",
            "value": 0.200872347,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.0831453,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default",
            "value": 0.000869206,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Julia",
            "value": 447.05275591000003,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default",
            "value": 0.002996862,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.210112577,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default",
            "value": 0.97579293,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Julia",
            "value": 288.21707239700004,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default",
            "value": 3.354548638,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default_manual_vectorized",
            "value": 1.705503447,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Julia",
            "value": 23.073395561,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.020505663,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Julia",
            "value": 15.87073237,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.004754824,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Julia",
            "value": 49.018954787000006,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Julia",
            "value": 15.132902325000002,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default",
            "value": 0.00776967,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 6.204340094,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.005839265,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.083370873,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.02529502,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.006030691,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default",
            "value": 0.021624086,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.007372383,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default",
            "value": 4.702892139,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.000580122,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.011024961,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000057655,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.010040369,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default",
            "value": 0.036250408,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003134244,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000025277,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000205716,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.028540149,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.001896256,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default",
            "value": 0.000444299,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000163406,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.020207759,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.006776079,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002812226,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.011671478,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.011146386,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000627473,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000439566,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000234577,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003329956,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007092235,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000262171,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.110381339,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040169824,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default",
            "value": 0.000293101,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default",
            "value": 0.000177073,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default_manual_vectorized",
            "value": 0.001874538,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.024353257,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007153053,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default",
            "value": 0.000282372,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.010961932,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.00312269,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default",
            "value": 0.013122485,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000107738,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007078812,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007193143,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default",
            "value": 0.001935628,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.00023666,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000254551,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default",
            "value": 0.000045028,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.001125052,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001471996,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default",
            "value": 0.000054785,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000054308,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default",
            "value": 0.000462446,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default",
            "value": 0.02046267,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.012421102,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.00191134,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default",
            "value": 0.020337181,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000261345,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.00073476,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000605214,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000465482,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003128434,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.0031211,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001915562,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default",
            "value": 0.000025905,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default",
            "value": 0.000106394,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.011142659,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000497893,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.000620203,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default",
            "value": 0.000106746,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default",
            "value": 0.00025569,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default",
            "value": 0.024870358,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.001067461,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000586844,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.00021768,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006302,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.021655941,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.0030933,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000031642,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default",
            "value": 0.000072483,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.001138465,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000016406,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000024127,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.001697647,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default",
            "value": 0.043055655,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000075071,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000952985,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004054315,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002020245,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004180629,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006104,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default_manual_vectorized",
            "value": 0.001092855,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.001697401,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default",
            "value": 0.001580306,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.00308452,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default",
            "value": 0.000045078,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default",
            "value": 0.000035595,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000058214,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default",
            "value": 0.01959729,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000047727,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default",
            "value": 0.26175369,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.0000374,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000023495,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000208188,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.001697541,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default",
            "value": 0.000016396,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default",
            "value": 0.009982612,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.001434886,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default",
            "value": 0.000030373,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000053243,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866739,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002959443,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929414,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017433,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.005707734,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.00309379,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027324,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004180766,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default",
            "value": 0.000087478,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default",
            "value": 0.000051592,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000087488,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.000207511,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default",
            "value": 0.000023495,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.027860268,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.022079384,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default",
            "value": 0.026676233,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000072641,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001697851,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default",
            "value": 0.0000866,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018255,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.00002724,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default",
            "value": 0.000024125,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004180413,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.000027464,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027322,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000045547,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.021540492,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.261753663,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default",
            "value": 0.000055838,
            "unit": "s"
          }
        ]
      },
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
          "id": "b6550c2cb6d24be4682eb2ca31b61e109f5377ae",
          "message": "Format Julia code (#2060)\n\nCo-authored-by: enzyme-ci-bot[bot] <78882869+enzyme-ci-bot[bot]@users.noreply.github.com>",
          "timestamp": "2026-01-02T16:59:36Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/b6550c2cb6d24be4682eb2ca31b61e109f5377ae"
        },
        "date": 1767414468342,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.414751849,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.676781132,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.425570402,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.066098285,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.00098215,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.11189551,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.000557685,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default",
            "value": 0.023650867,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.010754195,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.055345437,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Julia",
            "value": 39.621743322,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.024296251,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default_manual_vectorized",
            "value": 0.077257285,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.005323392,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 3.566791095,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default",
            "value": 0.030250137,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default",
            "value": 0.528996441,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Julia",
            "value": 0.058553056000000006,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.012126684,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.114417945,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/Default",
            "value": 0.105027954,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.007163757,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.020444729,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Julia",
            "value": 0.028394382000000003,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.009762941,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default",
            "value": 0.035814191,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Julia",
            "value": 23.131328401,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001450551,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default",
            "value": 0.000537737,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.614543476,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 3.621267801,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.027544154,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default",
            "value": 0.007378555,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.001569445,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default",
            "value": 0.908851283,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default_manual_vectorized",
            "value": 0.890671909,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.430299426,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.017115067,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.180060248,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Julia",
            "value": 67.635482159,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.009233481,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.435052468,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default",
            "value": 0.003711792,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Julia",
            "value": 0.040881104,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.004996692,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Julia",
            "value": 0.34856773700000004,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default",
            "value": 0.007094327,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Julia",
            "value": 1.6561050560000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.10085685900000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.195272091,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.431403173,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Julia",
            "value": 26.785266538000002,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default",
            "value": 0.031841206,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Julia",
            "value": 0.000565155,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.181395008,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Julia",
            "value": 0.20933779100000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.083359413,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default",
            "value": 0.000999401,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Julia",
            "value": 400.355437569,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default",
            "value": 0.001924886,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.193010333,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default",
            "value": 0.718103534,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Julia",
            "value": 274.016700814,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default",
            "value": 3.194040784,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default_manual_vectorized",
            "value": 1.532431718,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Julia",
            "value": 23.13988781,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.020684822,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAll",
            "value": 0.37588949,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Julia",
            "value": 15.623428800000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.003761608,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.01338685,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Julia",
            "value": 50.129324231000005,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Julia",
            "value": 11.230192160000001,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default",
            "value": 0.008842613,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/NoOpt",
            "value": 0.099531212,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 5.434621515,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/DisableTransposeReshape",
            "value": 0.107315953,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/NoOpt",
            "value": 0.412878788,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.005164462,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.079716728,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.021776335,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.005142633,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default",
            "value": 0.020883419,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.006874981,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default",
            "value": 4.290383851,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.000588536,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.011027549,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000059277,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.009981856,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/NoOpt",
            "value": 0.001252382,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default",
            "value": 0.036160029,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003131716,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000025347,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.00020616,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.027354423,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.001911982,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default",
            "value": 0.000446014,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/NoOpt",
            "value": 0.006851111,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.00015478,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.020722047,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.007091016,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002832934,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.011227867,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001252705,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.010955255,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003445302,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000588644,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000441621,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000221314,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003326625,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007123926,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003444999,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.108564556,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000259679,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.109026601,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.04028375,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default",
            "value": 0.000299314,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default",
            "value": 0.000155152,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default_manual_vectorized",
            "value": 0.001896965,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.024942974,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007181545,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default",
            "value": 0.000280613,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.011061755,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003117608,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.00321948,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAll",
            "value": 0.003895914,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default",
            "value": 0.013276132,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000108375,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007106409,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007219381,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default",
            "value": 0.002246426,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000222489,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/Default",
            "value": 0.000966534,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000255794,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default",
            "value": 0.000042656,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.001094027,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001409875,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default",
            "value": 0.000054836,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.006888943,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000054711,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default",
            "value": 0.000464149,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default",
            "value": 0.020985385,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.013265227,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.00192623,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default",
            "value": 0.020699883,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000257381,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.000700957,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.00059474,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000467666,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003056757,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003132411,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001927616,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default",
            "value": 0.000025738,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default",
            "value": 0.000106525,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.011006916,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000500231,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.006390111,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.000660781,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.001345511,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default",
            "value": 0.000107123,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default",
            "value": 0.000257215,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default",
            "value": 0.024793162,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.006742611,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.001051635,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000586128,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.000217856,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006269,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.020912725,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.003120805,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.005005183,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000031263,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default",
            "value": 0.000072405,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.001138312,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000016281,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000024179,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.00169891,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default",
            "value": 0.043055606,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000074893,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.00095279,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004052794,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.00202152,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.00418115,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006047,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default_manual_vectorized",
            "value": 0.001086193,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.001698123,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default",
            "value": 0.001649293,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/DisableTransposeReshape",
            "value": 0.002864862,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085193,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default",
            "value": 0.000045021,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default",
            "value": 0.000035594,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000058229,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/NoOpt",
            "value": 0.005226625,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default",
            "value": 0.019597128,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000047442,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default",
            "value": 0.261753691,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000018247,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000037297,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.00475239,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.00002343,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000207977,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.001698118,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default",
            "value": 0.000016284,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default",
            "value": 0.009982533,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004668181,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.001434632,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default",
            "value": 0.000030193,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000052963,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866964,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002959732,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000208264,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929436,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017381,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.00570715,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003121117,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027285,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004178626,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default",
            "value": 0.000087402,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default",
            "value": 0.000051671,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000087491,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.020824731,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.000207466,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default",
            "value": 0.000023458,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.02786009,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.022079284,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default",
            "value": 0.026676147,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000072455,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001698864,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default",
            "value": 0.000086372,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018237,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/Default",
            "value": 0.002349525,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027339,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005179581,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default",
            "value": 0.000024125,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/NoOpt",
            "value": 0.00286805,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAll",
            "value": 0.004744305,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004180242,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.000027481,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027409,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005004981,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000045592,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.020932942,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.261753665,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default",
            "value": 0.000055824,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]",
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "13771b008e590945717f986b9e55ab2f466959a6",
          "message": "Update EnzymeAD/Enzyme-JAX to commit a6af9b3e7ad5d56fde968895e2e420ef085062a2 (#2062)\n\nDiff: https://github.com/EnzymeAD/Enzyme-JAX/compare/2680f717edd6ea63b8433351e9dda18b4bdbf62b...a6af9b3e7ad5d56fde968895e2e420ef085062a2\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2026-01-04T01:26:44Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/13771b008e590945717f986b9e55ab2f466959a6"
        },
        "date": 1767500918732,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.384253794,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.602091419,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.301446874,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.067626309,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.000796839,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.100510334,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.000491703,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default",
            "value": 0.021775527,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.00808796,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.049988154,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Julia",
            "value": 37.955964582,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.020095942,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default_manual_vectorized",
            "value": 0.075229596,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.004538027,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 3.423210043,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default",
            "value": 0.02789805,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default",
            "value": 0.109440491,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Julia",
            "value": 0.067500224,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.009424651,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.104699801,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/Default",
            "value": 0.100976227,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.006843682,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.013696703,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Julia",
            "value": 0.026860547000000002,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.009905171,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default",
            "value": 0.038423109,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Julia",
            "value": 22.982550590000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001379819,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default",
            "value": 0.000543012,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.542406203,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 3.484970795,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.031543282,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default",
            "value": 0.00685052,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.001325607,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default",
            "value": 0.657794683,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default_manual_vectorized",
            "value": 0.649656439,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.406559452,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.015409049,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.164679295,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Julia",
            "value": 61.028672584000006,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.007862542,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.413312583,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default",
            "value": 0.003403642,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Julia",
            "value": 0.035574215,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.004789091,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Julia",
            "value": 0.353114804,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default",
            "value": 0.006353574,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Julia",
            "value": 1.646867906,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.08827246000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.173630587,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.406476378,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Julia",
            "value": 27.532541790000003,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default",
            "value": 0.029390514,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Julia",
            "value": 0.0005629890000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.163838356,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Julia",
            "value": 0.19435031500000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.072678947,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default",
            "value": 0.000951991,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Julia",
            "value": 394.239785004,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default",
            "value": 0.002425008,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.17527089,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default",
            "value": 0.638273506,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Julia",
            "value": 281.64813696100003,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default",
            "value": 2.235360891,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default_manual_vectorized",
            "value": 1.16410349,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Julia",
            "value": 22.947051448,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.019418325,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAll",
            "value": 0.353947181,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Julia",
            "value": 15.55157834,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.005591114,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.012671614,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Julia",
            "value": 44.676343954000004,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Julia",
            "value": 11.06896884,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default",
            "value": 0.007422852,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/NoOpt",
            "value": 0.09279552,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 4.992608701,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/DisableTransposeReshape",
            "value": 0.104123302,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/NoOpt",
            "value": 0.391028049,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.004780883,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.070056267,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.020087872,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.004465332,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default",
            "value": 0.020433787,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.006592783,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default",
            "value": 3.952025146,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.00057925,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.01159144,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000057139,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.011676024,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/NoOpt",
            "value": 0.001245298,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default",
            "value": 0.036163539,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003085847,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000025184,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000205757,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.0280182,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.001900969,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default",
            "value": 0.000447805,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/NoOpt",
            "value": 0.006835283,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000158416,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.019804514,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.007059677,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002751326,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.011173692,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001233986,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.010664612,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003431052,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.00061248,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000442013,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000222797,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003282652,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007108179,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003428369,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.108230603,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000255962,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.108713842,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040165534,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default",
            "value": 0.00028544,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default",
            "value": 0.000160864,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default_manual_vectorized",
            "value": 0.001896612,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.024526162,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007145449,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default",
            "value": 0.000277155,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.011222848,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003103544,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003217819,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAll",
            "value": 0.00390572,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default",
            "value": 0.013043807,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000108308,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007076223,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007186119,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default",
            "value": 0.001938155,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000229828,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/Default",
            "value": 0.000964143,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000254567,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default",
            "value": 0.000042356,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.001094435,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001390899,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default",
            "value": 0.000055223,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.006882725,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000054759,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default",
            "value": 0.000462193,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default",
            "value": 0.019963411,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.012342782,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.001911323,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default",
            "value": 0.020082231,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000257667,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.000700059,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000605046,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000465571,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003022125,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003099137,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001918908,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default",
            "value": 0.000024906,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default",
            "value": 0.000106096,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.010719679,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.00049822,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.006362553,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.000598556,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.001332676,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default",
            "value": 0.000107085,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default",
            "value": 0.000256077,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default",
            "value": 0.025280896,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.006989089,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.001046322,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.0005866,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.00021787,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.00000634,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.02100073,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.0030937,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.005005481,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.00003146,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default",
            "value": 0.000072565,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.00113844,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000016411,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000024223,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.001697453,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default",
            "value": 0.043055611,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000074946,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000952981,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004053246,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002020054,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004180404,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006176,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default_manual_vectorized",
            "value": 0.001086335,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.001697747,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default",
            "value": 0.001574189,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/DisableTransposeReshape",
            "value": 0.002864904,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085568,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default",
            "value": 0.000045106,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default",
            "value": 0.000035692,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000058021,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/NoOpt",
            "value": 0.005226972,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default",
            "value": 0.019597198,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000047652,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default",
            "value": 0.261753554,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000018161,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000037346,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004751319,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000023437,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000208131,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.001697253,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default",
            "value": 0.000016319,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default",
            "value": 0.009921898,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004668627,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.001433365,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default",
            "value": 0.000030329,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000053211,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866649,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002959883,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000208345,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929113,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017274,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.005707473,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003093202,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027195,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004179947,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default",
            "value": 0.000087401,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default",
            "value": 0.000051752,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000087492,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.020954375,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.000207489,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default",
            "value": 0.000023385,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.027859659,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.022079231,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default",
            "value": 0.026676048,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000072615,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001697169,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default",
            "value": 0.000086436,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018088,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/Default",
            "value": 0.002349644,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027236,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005179605,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default",
            "value": 0.000024126,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/NoOpt",
            "value": 0.002867705,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAll",
            "value": 0.004744813,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004180401,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.000027277,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027205,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005005221,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000045581,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.020835503,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.261753523,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default",
            "value": 0.000055792,
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
          "id": "12a0d7ea5cdc893488a3702cb83a31b187d1c83e",
          "message": "Update WORKSPACE",
          "timestamp": "2026-01-04T22:42:48Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/12a0d7ea5cdc893488a3702cb83a31b187d1c83e"
        },
        "date": 1767587968004,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.45267172,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.732602832,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.451946363,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.077290575,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.000982223,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.125469216,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.000585464,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default",
            "value": 19.091607714,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.011847053,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.063541557,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Julia",
            "value": 41.977156925,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.025222395,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default_manual_vectorized",
            "value": 0.078088287,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.006035707,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 3.663374202,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default",
            "value": 0.030295062,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default",
            "value": 0.116003151,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Julia",
            "value": 0.063112393,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.014189474,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.128326406,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/Default",
            "value": 0.112906964,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.006802601,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.03505063,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Julia",
            "value": 0.027650407000000002,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.010897558,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default",
            "value": 0.03339097,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Julia",
            "value": 23.003366308,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001785433,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default",
            "value": 0.000573243,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.674628785,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 3.726975923,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.042811608,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default",
            "value": 0.007856379,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.001847088,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default",
            "value": 1.011440862,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default_manual_vectorized",
            "value": 1.009787358,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.477623231,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.018498288,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.204329356,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Julia",
            "value": 70.573367278,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.009619261,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.460324129,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default",
            "value": 0.00430291,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Julia",
            "value": 0.051069911,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.00614401,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Julia",
            "value": 0.362379382,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default",
            "value": 0.006213508,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Julia",
            "value": 1.760934479,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.119873872,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.220948268,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.457979374,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Julia",
            "value": 26.957557654000002,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default",
            "value": 0.031530905,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Julia",
            "value": 0.000564124,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.204059535,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Julia",
            "value": 0.18016595900000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.085060486,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default",
            "value": 0.001002839,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Julia",
            "value": 398.26580808100005,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default",
            "value": 0.003159909,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.221313854,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default",
            "value": 1.894312897,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Julia",
            "value": 296.977279586,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default",
            "value": 3.943483493,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default_manual_vectorized",
            "value": 1.691256201,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Julia",
            "value": 22.930707468,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.02039845,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAll",
            "value": 0.416701061,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Julia",
            "value": 14.814461312,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.005117239,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.014839047,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Julia",
            "value": 38.697851756000006,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Julia",
            "value": 15.152773645000002,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default",
            "value": 0.006969623,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/NoOpt",
            "value": 0.104475993,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 6.221592769000001,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/DisableTransposeReshape",
            "value": 0.113207761,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/NoOpt",
            "value": 0.463821479,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.005810297,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.085721557,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.020730185,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.005776101,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default",
            "value": 7.752824972,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.007205354,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default",
            "value": 0.0453411,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.000590616,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.011037642,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000057213,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.009939551,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/NoOpt",
            "value": 0.001251759,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default",
            "value": 0.000498641,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003104907,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000025588,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000206172,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.027105082,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.001894838,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default",
            "value": 0.000444875,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/NoOpt",
            "value": 0.006843699,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000159593,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.020626216,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.00706609,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002805755,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.012409415,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001241875,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.011070488,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003439679,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000596663,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000442315,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000251865,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003310049,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007085158,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003435469,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.108243532,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000263759,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.108838014,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040210665,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default",
            "value": 0.00029038,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default",
            "value": 0.000156873,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default_manual_vectorized",
            "value": 0.001881737,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.024403233,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007129698,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default",
            "value": 0.007849405,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.010990006,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003125733,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.00321548,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAll",
            "value": 0.003927211,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default",
            "value": 0.013033427,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000107075,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007068503,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.00719592,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default",
            "value": 0.001945208,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000227599,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/Default",
            "value": 0.000964668,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000255551,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default",
            "value": 0.000042596,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.001247389,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001406674,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default",
            "value": 0.000054387,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.006902346,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000054429,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default",
            "value": 0.000462173,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default",
            "value": 0.026327856,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.012449159,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.001909586,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default",
            "value": 0.027530982,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000258213,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.000728957,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000604,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000465604,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003052229,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003118566,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001911464,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default",
            "value": 0.000025856,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default",
            "value": 0.000106147,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.011024945,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000498692,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.006364617,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.000619053,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.001337633,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default",
            "value": 0.000109027,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default",
            "value": 0.007683661,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default",
            "value": 0.023907813,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.006985016,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.001067514,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000587222,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.000217758,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006291,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.021094026,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.003093394,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.005005075,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000031091,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default",
            "value": 0.000072697,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.001138527,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000016408,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000024143,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.001699349,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default",
            "value": 0.000058496,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000075001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000953076,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004055209,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002022162,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004181267,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006059,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default_manual_vectorized",
            "value": 0.001086528,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.001699076,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default",
            "value": 0.001582191,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/DisableTransposeReshape",
            "value": 0.002865266,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085568,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default",
            "value": 0.000045089,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default",
            "value": 0.000036462,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000057977,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/NoOpt",
            "value": 0.005226458,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default",
            "value": 0.019497517,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000047682,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default",
            "value": 0.261753759,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000018249,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000037398,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004752578,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000023447,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000208368,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.001698022,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default",
            "value": 0.000016366,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default",
            "value": 0.008142611,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004670515,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.001438288,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default",
            "value": 0.010515783,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000053133,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866737,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002959708,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000208486,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929664,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017423,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.005708068,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003093368,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.00002742,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004180769,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default",
            "value": 0.000087454,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default",
            "value": 0.010489052,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000087409,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.0211158,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.000207722,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default",
            "value": 0.000023437,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.02711748,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.022079374,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default",
            "value": 0.029050345,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000072642,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001698767,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default",
            "value": 0.000086721,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018213,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/Default",
            "value": 0.002349451,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027439,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005179952,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default",
            "value": 0.000024185,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/NoOpt",
            "value": 0.002867604,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAll",
            "value": 0.004743864,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004181415,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.000027598,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027418,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005005098,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000045573,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.021107549,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.261753763,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default",
            "value": 0.000055853,
            "unit": "s"
          }
        ]
      },
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
          "id": "d86ca12aee2ad225aa0617d98b715f7360b7b9e5",
          "message": "Regenerate MLIR Bindings (#2076)\n\nCo-authored-by: enzyme-ci-bot[bot] <78882869+enzyme-ci-bot[bot]@users.noreply.github.com>",
          "timestamp": "2026-01-06T03:13:01Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/d86ca12aee2ad225aa0617d98b715f7360b7b9e5"
        },
        "date": 1767673931047,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.430394479,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.914031985,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.441252555,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.068851118,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.000946027,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.12190159,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.000549523,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default",
            "value": 19.071585586,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.011753286,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.06175789,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Julia",
            "value": 39.077789528000004,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.022491328,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default_manual_vectorized",
            "value": 0.080590964,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.005655516,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 3.609616283,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default",
            "value": 0.028839564,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default",
            "value": 0.117902699,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Julia",
            "value": 0.06049663500000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.013916234,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.126606508,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/Default",
            "value": 0.109861783,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.006600724,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.030573587,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Julia",
            "value": 0.030647201000000002,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.010512206,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default",
            "value": 0.037579023,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Julia",
            "value": 23.108849667,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001833639,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default",
            "value": 0.000593819,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.623338807,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 3.724168541,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.043450721,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default",
            "value": 0.007422281,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.001717199,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default",
            "value": 0.978501197,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default_manual_vectorized",
            "value": 0.965124529,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.453826156,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.018190983,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.193129107,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Julia",
            "value": 66.886500002,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.007114292,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.464519693,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default",
            "value": 0.003953573,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Julia",
            "value": 0.035544531000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.00553971,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Julia",
            "value": 0.351524449,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default",
            "value": 0.006490875,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Julia",
            "value": 1.6553976300000002,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.10835281000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.205515963,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.450768939,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Julia",
            "value": 29.619023076,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default",
            "value": 0.032365505,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Julia",
            "value": 0.000567186,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.189400373,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Julia",
            "value": 0.18377451,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.081868633,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default",
            "value": 0.000909406,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Julia",
            "value": 285.72305800000004,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default",
            "value": 0.003150132,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.204924793,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default",
            "value": 1.706631665,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Julia",
            "value": 294.355842601,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default",
            "value": 3.85173703,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default_manual_vectorized",
            "value": 1.650522015,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Julia",
            "value": 23.127872573,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.02083369,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAll",
            "value": 0.39510534,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Julia",
            "value": 15.415740167000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.004960246000000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.014701198,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Julia",
            "value": 44.02594672,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Julia",
            "value": 13.769516616,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default",
            "value": 0.007346677,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/NoOpt",
            "value": 0.100120565,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 5.9693329230000005,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/DisableTransposeReshape",
            "value": 0.110674358,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/NoOpt",
            "value": 0.438830023,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.005494478,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.080772318,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.026109021,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.005950238,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default",
            "value": 7.600093442,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.007432898,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default",
            "value": 0.045842976,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.000573146,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.01068908,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000058535,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.009642591,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/NoOpt",
            "value": 0.001253466,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default",
            "value": 0.000500471,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003081114,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000025399,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000206979,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.028095186,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.001899628,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default",
            "value": 0.000445027,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/NoOpt",
            "value": 0.006852932,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000157536,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.020250871,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.00676504,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002753916,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.011059442,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001244252,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.010740286,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003456821,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000857152,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000442906,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000229219,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003264269,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007112222,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003449313,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.108494217,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000264512,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.108891163,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040159631,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default",
            "value": 0.00030199,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default",
            "value": 0.000155597,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default_manual_vectorized",
            "value": 0.001475984,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.023979125,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007175436,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default",
            "value": 0.007895775,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.010674573,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003091831,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003226065,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAll",
            "value": 0.003913802,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default",
            "value": 0.013216157,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000106848,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007098429,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007218748,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default",
            "value": 0.002661073,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000213623,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/Default",
            "value": 0.000967318,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000256256,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default",
            "value": 0.000044743,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.001081635,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.00140612,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default",
            "value": 0.000054333,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.006910027,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000054679,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default",
            "value": 0.000464118,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default",
            "value": 0.020371469,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.013213058,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.001913494,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default",
            "value": 0.029549654,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000260583,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.000694963,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000573846,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000467348,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003007896,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003083627,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001922041,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default",
            "value": 0.000025594,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default",
            "value": 0.000106579,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.010699743,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000500012,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.006382707,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.00057614,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.001347722,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default",
            "value": 0.000107079,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default",
            "value": 0.007705689,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default",
            "value": 0.023638216,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.007016582,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.001037114,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000585467,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.000217715,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006311,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.020638798,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.003093055,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.005005034,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000031046,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default",
            "value": 0.000072512,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.00113854,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000016366,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000024204,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.00169728,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default",
            "value": 0.00005836,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000074995,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000953114,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004053636,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002019788,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.00418055,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006122,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default_manual_vectorized",
            "value": 0.001086769,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.001697563,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default",
            "value": 0.001650278,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/DisableTransposeReshape",
            "value": 0.002864936,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085283,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default",
            "value": 0.000045079,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default",
            "value": 0.000036496,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000057884,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/NoOpt",
            "value": 0.005226658,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default",
            "value": 0.019497458,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000047523,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default",
            "value": 0.26175353,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000018163,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000037514,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004751407,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.00002344,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000208147,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.001696972,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default",
            "value": 0.000016318,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default",
            "value": 0.008141326,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004670416,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.001436077,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default",
            "value": 0.010515686,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.00005323,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866812,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002958758,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000208439,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.00092931,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017354,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.005707562,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003093187,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027293,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004180316,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default",
            "value": 0.000087425,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default",
            "value": 0.010488998,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000087391,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.020576909,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.000207532,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default",
            "value": 0.000023446,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.027117353,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.02207934,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default",
            "value": 0.029050331,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.0000725,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001697576,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default",
            "value": 0.000086447,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018141,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/Default",
            "value": 0.002349326,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027162,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005180181,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default",
            "value": 0.000024112,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/NoOpt",
            "value": 0.002867975,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAll",
            "value": 0.004743717,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004179804,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.000027319,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027249,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005005319,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000045522,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.020569565,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.261753476,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default",
            "value": 0.000055767,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]",
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "1ab54596e6f2c446b8f1b6cb9e33bc37f635de3a",
          "message": "Update EnzymeAD/Enzyme-JAX to commit a83eae480f2c9450ad774e9432fad222c33a5411 (#2081)\n\nDiff: https://github.com/EnzymeAD/Enzyme-JAX/compare/418f80e35da49a0c2a3e7f5599f21dc062b5d5b8...a83eae480f2c9450ad774e9432fad222c33a5411\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2026-01-06T20:30:36Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/1ab54596e6f2c446b8f1b6cb9e33bc37f635de3a"
        },
        "date": 1767761347540,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.594841988,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 1.300704505,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.908699022,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.095595714,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.001616218,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.159534354,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.000807448,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default",
            "value": 22.112040597,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.011459126,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.080138509,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Julia",
            "value": 67.339393058,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.04208987,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default_manual_vectorized",
            "value": 0.101451412,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.006822045,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 4.264414001,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default",
            "value": 0.037363376,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default",
            "value": 0.958261019,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Julia",
            "value": 0.125150163,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.014326966,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.161516695,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/Default",
            "value": 0.134247135,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.009185677,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.053384092,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Julia",
            "value": 0.083087385,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.013422184,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default",
            "value": 0.045077267,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Julia",
            "value": 23.165313138000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.002386416,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default",
            "value": 0.000827769,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.826724913,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 4.310812556,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.058389657,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default",
            "value": 0.011093441,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.002360977,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default",
            "value": 1.263658472,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default_manual_vectorized",
            "value": 1.313223271,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.607988186,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.027216767,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.254337174,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Julia",
            "value": 121.477363135,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.009781024,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.594686034,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default",
            "value": 0.005360252,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Julia",
            "value": 0.067021762,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.006547064,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Julia",
            "value": 0.38797048700000003,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default",
            "value": 0.008290502,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Julia",
            "value": 2.240151429,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.13627619800000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.268803727,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.603863108,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Julia",
            "value": 29.664934517000003,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default",
            "value": 0.040888423,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Julia",
            "value": 0.000666749,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.251430318,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Julia",
            "value": 0.204636917,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.108778268,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default",
            "value": 0.001664771,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Julia",
            "value": 451.281854633,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default",
            "value": 0.003691254,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.27007125,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default",
            "value": 2.631180919,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Julia",
            "value": 317.762573824,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default",
            "value": 4.741101239,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default_manual_vectorized",
            "value": 2.443262331,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Julia",
            "value": 23.183309975,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.026436905,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAll",
            "value": 0.519613794,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Julia",
            "value": 19.71638585,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.006235058000000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.015450488,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Julia",
            "value": 69.23340861,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Julia",
            "value": 15.512556003,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default",
            "value": 0.009871041,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/NoOpt",
            "value": 0.12857645,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 6.777950189,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/DisableTransposeReshape",
            "value": 0.141425533,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/NoOpt",
            "value": 0.585984685,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.006449198,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.107029141,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.031398677,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.006645256,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default",
            "value": 18.381791,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.011241758,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default",
            "value": 0.057750035,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.00057684,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.010858782,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000058365,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.010099134,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/NoOpt",
            "value": 0.001258751,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default",
            "value": 0.000501216,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003147687,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000025914,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000207331,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.027913593,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.00191138,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default",
            "value": 0.000450877,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/NoOpt",
            "value": 0.006883157,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000161383,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.020256467,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.007082374,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002870451,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.011732374,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001257615,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.010941838,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003460481,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000605699,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000444572,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000233981,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003350646,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.00714059,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003456759,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.108862493,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000265735,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.1093241,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040377514,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default",
            "value": 0.000295805,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default",
            "value": 0.000171229,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default_manual_vectorized",
            "value": 0.00188549,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.024661151,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007206291,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default",
            "value": 0.007871589,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.011034436,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003244956,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003247964,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAll",
            "value": 0.003918757,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default",
            "value": 0.013314474,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000112007,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.00712925,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.00724733,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default",
            "value": 0.00266571,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000229002,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/Default",
            "value": 0.000978403,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000256215,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default",
            "value": 0.000048576,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.00115309,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001424344,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default",
            "value": 0.000054801,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.006909001,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000055354,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default",
            "value": 0.000464819,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default",
            "value": 0.020351085,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.013310819,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.001946726,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default",
            "value": 0.027170919,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000263891,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.000761801,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000601328,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000467952,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003095815,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.00325027,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001941088,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default",
            "value": 0.000026503,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default",
            "value": 0.000106845,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.01113392,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000500998,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.006398781,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.000602889,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.001361539,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default",
            "value": 0.000112103,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default",
            "value": 0.007706845,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default",
            "value": 0.024324777,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.006403787,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.001121391,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000585826,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.000217881,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006198,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.020580223,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.0030931,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.005005186,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000031125,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default",
            "value": 0.000072542,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.001138122,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000016399,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000024106,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.001697108,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default",
            "value": 0.00005832,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000075003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000953189,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004053918,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002020481,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004180861,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.00000613,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default_manual_vectorized",
            "value": 0.001086274,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.001697284,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default",
            "value": 0.001649471,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/DisableTransposeReshape",
            "value": 0.002864798,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085359,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default",
            "value": 0.000045071,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default",
            "value": 0.000036429,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000057887,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/NoOpt",
            "value": 0.005227336,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default",
            "value": 0.019497543,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000047588,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default",
            "value": 0.261753488,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.00001818,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000037326,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004751371,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000023495,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000208142,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.001696853,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default",
            "value": 0.000016283,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default",
            "value": 0.008142885,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.00467111,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.001437427,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default",
            "value": 0.010515712,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000053146,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866744,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.00295924,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000208376,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929362,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.00001731,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.005707807,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003093864,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027181,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004180632,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default",
            "value": 0.000087479,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default",
            "value": 0.010489092,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000087389,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.020625049,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.000207568,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default",
            "value": 0.000023434,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.027117265,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.022079391,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default",
            "value": 0.029050421,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.00007252,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.00169668,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default",
            "value": 0.000086465,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018157,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/Default",
            "value": 0.00234915,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027191,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005180536,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default",
            "value": 0.000024109,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/NoOpt",
            "value": 0.002867723,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAll",
            "value": 0.00474512,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004180436,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.000027355,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027209,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00500498,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000045559,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.020582046,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.261753557,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default",
            "value": 0.00005573,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]",
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "0a7db58a9e713b9526a0caa42c4c70c425c25cf8",
          "message": "Update EnzymeAD/Enzyme-JAX to commit cde53513cfe2ba1af09485fc02860a2e5f372d98 (#2086)\n\nDiff: https://github.com/EnzymeAD/Enzyme-JAX/compare/6e3b26632853058ec79d8415637483ba8b5b71b5...cde53513cfe2ba1af09485fc02860a2e5f372d98\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2026-01-08T04:32:53Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/0a7db58a9e713b9526a0caa42c4c70c425c25cf8"
        },
        "date": 1767846861835,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.434987494,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.6205963,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 2.427018648,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.073717599,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.000969686,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.121640885,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.000566044,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default",
            "value": 18.701393584,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.011422281,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.061344144,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Julia",
            "value": 37.211384196000004,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.022568905,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default_manual_vectorized",
            "value": 0.078604194,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.00568051,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 3.597980695,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default",
            "value": 0.028642859,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default",
            "value": 0.115032024,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Julia",
            "value": 0.064943794,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.014097495,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.126510459,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/Default",
            "value": 0.106747571,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.006795827,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.033326957,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Julia",
            "value": 0.028934500000000002,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.010344044,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default",
            "value": 0.037235667,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Julia",
            "value": 22.997137062,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.001724169,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default",
            "value": 0.000507614,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.598965332,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 3.699194863,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.042009379,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default",
            "value": 0.007199316,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.001740363,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default",
            "value": 0.914891326,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default_manual_vectorized",
            "value": 0.726096029,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.435557954,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.018254905,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.19081143,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Julia",
            "value": 66.908660528,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.009043011,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.448070121,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default",
            "value": 0.003907781,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Julia",
            "value": 0.033762772,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.005768315,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Julia",
            "value": 0.34206452200000004,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default",
            "value": 0.006186709,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Julia",
            "value": 1.65068872,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.12356728,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.205797538,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.440138585,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Julia",
            "value": 26.97483154,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default",
            "value": 0.030473465,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Julia",
            "value": 0.000568796,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.191061223,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Julia",
            "value": 0.187982277,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.078734515,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default",
            "value": 0.000963403,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Julia",
            "value": 398.14996388000003,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default",
            "value": 0.003102904,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.204549667,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default",
            "value": 1.550184455,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Julia",
            "value": 285.710478586,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default",
            "value": 3.513423071,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default_manual_vectorized",
            "value": 1.516548947,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Julia",
            "value": 23.069393021,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.02034975,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAll",
            "value": 0.399139788,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Julia",
            "value": 15.094610303000001,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.004735973,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.014203747,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Julia",
            "value": 40.711799094,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Julia",
            "value": 10.334452348000001,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default",
            "value": 0.008962647,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/NoOpt",
            "value": 0.100075232,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 5.459266406,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/DisableTransposeReshape",
            "value": 0.109703744,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/NoOpt",
            "value": 0.437824139,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.005590711,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.080889561,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.021429658,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.005702706,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default",
            "value": 7.656049554,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.007101052,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default",
            "value": 0.046944291,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.000740369,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.011185003,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000057709,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.010179396,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/NoOpt",
            "value": 0.001259651,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default",
            "value": 0.000499291,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003145738,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000027604,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000207129,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.028172295,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.001912156,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default",
            "value": 0.000449194,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/NoOpt",
            "value": 0.006854715,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000165507,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.020354177,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.007074144,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002896864,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.011933769,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001248283,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.011177619,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003456159,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.000606486,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000441038,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000240214,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003346543,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007110368,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003455204,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.108077783,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000269041,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.10857952,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040113822,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default",
            "value": 0.000289273,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default",
            "value": 0.000158175,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default_manual_vectorized",
            "value": 0.001468054,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.024281489,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007174535,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default",
            "value": 0.0078334,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.011242115,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003157599,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003218616,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAll",
            "value": 0.00389925,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default",
            "value": 0.013272297,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000110057,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007094844,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007220931,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default",
            "value": 0.001936735,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000239679,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/Default",
            "value": 0.000974645,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.00025462,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default",
            "value": 0.000043863,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.00113721,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001423517,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default",
            "value": 0.000053719,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.006892359,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000054109,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default",
            "value": 0.000462081,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default",
            "value": 0.020646755,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.013124384,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.001936938,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default",
            "value": 0.027210798,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000261135,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.000771308,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000608365,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000465284,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003159362,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003150814,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001936934,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default",
            "value": 0.000025732,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default",
            "value": 0.000106592,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.011379533,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000498013,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00637831,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.000642916,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.001358569,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default",
            "value": 0.000109945,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default",
            "value": 0.007668661,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default",
            "value": 0.025194895,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.006982877,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.001094378,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000586305,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.000217868,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006257,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.020610228,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.003093109,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.005005007,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000031181,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default",
            "value": 0.000072554,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.001138434,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000016361,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000024131,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.001697044,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default",
            "value": 0.000058349,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000074988,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.00095309,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004052567,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002020071,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004180267,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006051,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default_manual_vectorized",
            "value": 0.001086317,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.001696968,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default",
            "value": 0.001578689,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/DisableTransposeReshape",
            "value": 0.002865086,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085297,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default",
            "value": 0.000045079,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default",
            "value": 0.000036462,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000057879,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/NoOpt",
            "value": 0.005226775,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default",
            "value": 0.019497598,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000047558,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default",
            "value": 0.261753677,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000018161,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.00003725,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004751255,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000023385,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000208208,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.001696772,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default",
            "value": 0.000016307,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default",
            "value": 0.008135803,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.00467026,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.001436144,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default",
            "value": 0.01051575,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000053147,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866784,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002959448,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000208422,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000929318,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017305,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.005707423,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.00309343,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027288,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004179909,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default",
            "value": 0.000087481,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default",
            "value": 0.010489093,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000087464,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.020644359,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.000207558,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default",
            "value": 0.000023445,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.027117312,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.022079417,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default",
            "value": 0.029050388,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000072615,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.00169734,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default",
            "value": 0.000086447,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.000018137,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/Default",
            "value": 0.002349427,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027224,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005179394,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default",
            "value": 0.000024277,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/NoOpt",
            "value": 0.002868085,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAll",
            "value": 0.004743838,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004181571,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.00002733,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027306,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00500525,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.00004551,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.020523436,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.261753696,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default",
            "value": 0.000055753,
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
          "id": "6a952c1b5ffd49b0860e0125030568e4252e3e1e",
          "message": "Reduce window wrap (#2090)\n\n* Reduce window wrap\n\n* Update Project.toml",
          "timestamp": "2026-01-08T20:34:28Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/6a952c1b5ffd49b0860e0125030568e4252e3e1e"
        },
        "date": 1767907969027,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.561211427,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.73957874,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Default",
            "value": 3.292438486,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.107814797,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.001235729,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors",
            "value": 0.134081402,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.000621914,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default",
            "value": 0.074357575,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Default",
            "value": 0.008443759,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Default",
            "value": 0.063281568,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Julia",
            "value": 88.03587242,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.02698358,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default_manual_vectorized",
            "value": 0.104127322,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAll",
            "value": 0.00550727,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors",
            "value": 4.554789692,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Default",
            "value": 0.021643099,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Default",
            "value": 0.161210768,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Julia",
            "value": 0.135509133,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors",
            "value": 0.011101856,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.148826784,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/Default",
            "value": 0.133684909,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default_manual_vectorized",
            "value": 0.009096337,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.031841452,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Julia",
            "value": 0.09119691,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.011705896,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default",
            "value": 0.050239381,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Julia",
            "value": 23.245389122000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/Default",
            "value": 0.002116278,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CPU/Default",
            "value": 0.000709535,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.787691429,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 4.689865746,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.037795673,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default",
            "value": 0.008422127,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default",
            "value": 0.827937843,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Default_manual_vectorized",
            "value": 0.849046555,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.56712354,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.0289305,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.224504316,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CPU/Julia",
            "value": 170.565299054,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.010213374,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.570452439,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Default",
            "value": 0.004865775,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CPU/Julia",
            "value": 0.09482496300000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultBeforeEnzyme",
            "value": 0.005476569,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Julia",
            "value": 0.521506037,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Default",
            "value": 0.008770727,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Julia",
            "value": 2.300781443,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CPU/Julia",
            "value": 0.136837437,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.240515823,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DisableTransposeReshapeAll",
            "value": 0.556666832,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Julia",
            "value": 37.197588321000005,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/NoOpt",
            "value": 0.562972981,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/NoOpt",
            "value": 0.225402012,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default",
            "value": 0.022929341,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CPU/Julia",
            "value": 0.000672515,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Julia",
            "value": 0.21381213000000002,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CPU/Default",
            "value": 0.001292289,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CPU/Julia",
            "value": 358.864746367,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CPU/Default",
            "value": 0.002622335,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CPU/DefaultAll",
            "value": 0.246784311,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CPU/Default",
            "value": 0.780209469,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CPU/NoOpt",
            "value": 0.002112247,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Julia",
            "value": 421.36685864900005,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default",
            "value": 0.924776803,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/NoOpt",
            "value": 0.126173701,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CPU/Default_manual_vectorized",
            "value": 2.328769816,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CPU/Julia",
            "value": 23.226198670000002,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default_manual_vectorized",
            "value": 0.024326821,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/NoOpt",
            "value": 0.005437635,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CPU/DefaultAll",
            "value": 0.497933866,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Julia",
            "value": 26.641658431000003,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/Julia",
            "value": 0.004477056,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CPU/StructuredTensors (Only Detection)",
            "value": 0.013624416,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Julia",
            "value": 94.36347805000001,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CPU/Julia",
            "value": 10.806071712000001,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CPU/Default",
            "value": 0.009813352,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CPU/Julia",
            "value": 8.40181591,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/NoOpt",
            "value": 0.104260012,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CPU/DisableTransposeReshape",
            "value": 0.135862258,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CPU/DefaultAfterEnzyme",
            "value": 0.005249533,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CPU/Default",
            "value": 0.100097353,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.028838233,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CPU/Default",
            "value": 0.011086717,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CPU/Default_manual_vectorized",
            "value": 0.008308334,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CPU/Default",
            "value": 0.026565705,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/Default",
            "value": 0.000590489,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherPadAll",
            "value": 0.011274485,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000060071,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.010266028,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default",
            "value": 0.000500172,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAll",
            "value": 0.003167951,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000025876,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000206467,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.028454062,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/Default",
            "value": 0.001906748,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default",
            "value": 0.00044997,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.003198271,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000172138,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.020839249,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors",
            "value": 0.006961512,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/CUDA/Default",
            "value": 0.002950738,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/DisableTransposeReshape",
            "value": 0.001257482,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisableScatterGatherAll",
            "value": 0.011912538,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003451706,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAll",
            "value": 0.00064479,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000443241,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/Default",
            "value": 0.000252022,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeAll",
            "value": 0.003443295,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.007111763,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.10881638,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000272086,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/StructuredTensors",
            "value": 0.109256355,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/CUDA/Default",
            "value": 0.040234941,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default",
            "value": 0.000296845,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/CUDA/Default",
            "value": 0.000162672,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default_manual_vectorized",
            "value": 0.001889752,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default_manual_vectorized",
            "value": 0.025245935,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.007177348,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/NoOpt",
            "value": 0.001254553,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/CUDA/Default",
            "value": 0.000307776,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DisablePadAll",
            "value": 0.011672902,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.007224596,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003172936,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.003220226,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultAll",
            "value": 0.003904633,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default",
            "value": 0.013246676,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000111597,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.007099838,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/CUDA/Default",
            "value": 0.002247678,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/CUDA/Default",
            "value": 0.000989057,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/CUDA/NoOpt",
            "value": 0.00193454,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/NoOpt",
            "value": 0.000747314,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000255177,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/CUDA/Default",
            "value": 0.000050717,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors",
            "value": 0.001418382,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/NoOpt",
            "value": 0.00686652,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default",
            "value": 0.000054448,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.006906328,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000055162,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default",
            "value": 0.000463756,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/CUDA/Default",
            "value": 0.021026118,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/CUDA/Default_manual_vectorized",
            "value": 0.013241884,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/CUDA/Default",
            "value": 0.02074975,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000261344,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.000638748,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000466668,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/NoOpt",
            "value": 0.011959426,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/DefaultBeforeEnzyme",
            "value": 0.003168441,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/Default",
            "value": 0.001935448,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/CUDA/Default",
            "value": 0.000026189,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/CUDA/Default",
            "value": 0.000106445,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/CUDA/DefaultAll",
            "value": 0.011453827,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/CUDA/Default_manual_vectorized",
            "value": 0.000500067,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.006392195,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/CUDA/NoOpt",
            "value": 0.003357272,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/NoOpt",
            "value": 0.001182768,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/CUDA/DefaultAfterEnzyme",
            "value": 0.00065076,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.001358411,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/CUDA/NoOpt",
            "value": 0.000243821,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/CUDA/Default",
            "value": 0.000111383,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/CUDA/Default",
            "value": 0.00025632,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/CUDA/Default",
            "value": 0.025286825,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/CUDA/StructuredTensors (Only Detection)",
            "value": 0.007016319,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/CUDA/Default",
            "value": 0.001120801,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/Default",
            "value": 0.000217819,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/Default",
            "value": 0.021195025,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAll",
            "value": 0.003100122,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.005005781,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000031421,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000585472,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/NoOpt",
            "value": 0.001138387,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default",
            "value": 0.000072407,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000016181,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000024148,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherAll",
            "value": 0.001696009,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default",
            "value": 0.000058003,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000074559,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/primal/TPU/Default",
            "value": 0.000952842,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004179849,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/Default",
            "value": 0.000006002,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default_manual_vectorized",
            "value": 0.001086971,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableScatterGatherPadAll",
            "value": 0.001696056,
            "unit": "s"
          },
          {
            "name": "doitgen [256, 1024, 512]/primal/TPU/Default",
            "value": 0.001649089,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/DisableTransposeReshape",
            "value": 0.002865015,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.003085365,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default",
            "value": 0.000045053,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default",
            "value": 0.000036271,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/NoOpt",
            "value": 0.002959487,
            "unit": "s"
          },
          {
            "name": "syr2k [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000057628,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default",
            "value": 0.018832993,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.00004741,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default",
            "value": 0.261753606,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000018073,
            "unit": "s"
          },
          {
            "name": "gemmver [2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000037476,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.004052016,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.004751071,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000023492,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors",
            "value": 0.000207904,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisablePadAll",
            "value": 0.00169603,
            "unit": "s"
          },
          {
            "name": "3mm [256, 1024, 2048, 4096]/primal/TPU/Default",
            "value": 0.000016201,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default",
            "value": 0.00813759,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004670043,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DisableTransposeReshapeAll",
            "value": 0.001434855,
            "unit": "s"
          },
          {
            "name": "syrk [2048]/primal/TPU/Default",
            "value": 0.000030257,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default_manual_vectorized",
            "value": 0.000052973,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/NoOpt",
            "value": 0.002867723,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.000208114,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/Default",
            "value": 0.000928974,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/Default",
            "value": 0.000017234,
            "unit": "s"
          },
          {
            "name": "jacobi_1d [2048, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.005735415,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.003093189,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultBeforeEnzyme",
            "value": 0.000027154,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.004179804,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default",
            "value": 0.000087396,
            "unit": "s"
          },
          {
            "name": "covariance [2048, 2048]/primal/TPU/Default",
            "value": 0.000061407,
            "unit": "s"
          },
          {
            "name": "gesummv [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000087493,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/NoOpt",
            "value": 0.002018736,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/NoOpt",
            "value": 0.005226258,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [1024 x 1024]/primal/TPU/Default",
            "value": 0.000207346,
            "unit": "s"
          },
          {
            "name": "bicg [2048, 4096]/primal/TPU/Default",
            "value": 0.000023493,
            "unit": "s"
          },
          {
            "name": "fdtd_2d [1024, 2048, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.027117363,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default_manual_vectorized",
            "value": 0.022079296,
            "unit": "s"
          },
          {
            "name": "jacobi_2d [512, 512, 1024]/primal/TPU/Default",
            "value": 0.026676078,
            "unit": "s"
          },
          {
            "name": "gemm [2048, 4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000072426,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.001696653,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors (Only Detection)",
            "value": 0.021056524,
            "unit": "s"
          },
          {
            "name": "2mm [2048]/primal/TPU/Default",
            "value": 0.000086415,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [256 x 256]/primal/TPU/StructuredTensors",
            "value": 0.00001806,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/primal/TPU/Default",
            "value": 0.002349176,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAfterEnzyme",
            "value": 0.000027078,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/primal/TPU/NoOpt",
            "value": 0.000006263,
            "unit": "s"
          },
          {
            "name": "atax [2048]/primal/TPU/Default",
            "value": 0.000024099,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005179224,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DefaultAll",
            "value": 0.00474592,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/NoOpt",
            "value": 0.000027174,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/reverse/TPU/DefaultAll",
            "value": 0.004178966,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/reverse/TPU/DefaultAll",
            "value": 0.000027116,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/primal/TPU/NoOpt",
            "value": 0.000866646,
            "unit": "s"
          },
          {
            "name": "DGCNN [3, 128, 256]/reverse/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005005398,
            "unit": "s"
          },
          {
            "name": "mvt [4096]/primal/TPU/Default_manual_vectorized",
            "value": 0.000045456,
            "unit": "s"
          },
          {
            "name": "NewtonSchulz [4096 x 4096]/primal/TPU/StructuredTensors",
            "value": 0.021123542,
            "unit": "s"
          },
          {
            "name": "heat_3d [128, 128, 128, 256]/primal/TPU/Default_manual_vectorized",
            "value": 0.261753579,
            "unit": "s"
          },
          {
            "name": "correlation [2048, 2048]/primal/TPU/Default",
            "value": 0.000055585,
            "unit": "s"
          }
        ]
      }
    ]
  }
}