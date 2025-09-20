window.BENCHMARK_DATA = {
  "lastUpdate": 1758253087356,
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
          "id": "05eb248966be8af7e8e8f8e3a2f6371e63e18352",
          "message": "Regenerate MLIR Bindings (#1671)\n\nCo-authored-by: enzyme-ci-bot[bot] <78882869+enzyme-ci-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-09-17T03:09:21Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/05eb248966be8af7e8e8f8e3a2f6371e63e18352"
        },
        "date": 1758089008842,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.001958951,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.0016623940000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.004259335,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.004205484000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004122463000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.004066397,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.004172928,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.004232076,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.001809239,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.00468177,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004107095,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.0019297860000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.004238958,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004193769,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.004163511,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004121126,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004061887,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.0018715530000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.004259178000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.004108377,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.001955079,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004278897,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.00200564,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000642656,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.000728579,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0071185530000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0006824540000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.002940021,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002007937,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0072071570000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.002974111,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.000410848,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.002967514,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029257500000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.0011196180000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007213753000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007236846000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007109098,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001125437,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006546060000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.00031289800000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003078112,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.003075654,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0006515070000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.010415061000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.002893999,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007127156000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.000304416,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.00116813,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.002925219,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0032865200000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007201057,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0030507670000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003069023,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007091224,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007139774000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.000301792,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0031532860000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0006494210000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000659569,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007093876000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0006805330000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007135010000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0030914920000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007283109,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002000159,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030421940000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.00065724,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002912371,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007096149,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006531110000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001162065,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002451678,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002065888,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007141995000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029596040000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.0030917170000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0006850140000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.00110805,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007120179000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000648938,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.000646803,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002925165,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029099060000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0006544210000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.002002929,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.0010591160000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000301717,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001991578,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010519696,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000701469,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000314803,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029099670000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.002955493,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007208114000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000673093,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00463228,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00115902,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0029930100000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.00219985,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00031797,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00032972000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00286227,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0030078300000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.004607341,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0003187,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00015974,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.004613310000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0011471600000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0029734500000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00032681,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00032378,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0004381,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0029588500000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00298088,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00462451,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00110642,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00298109,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00076762,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00245163,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00114801,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00016001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00033140000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.004608919,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00016193000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00114595,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00299147,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00043397000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00033014000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00016414,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00033468,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00463764,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0029756300000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00031771,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00462409,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00462407,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00462888,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0028533910000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004609421000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0046260400000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00131049,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00044280000000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.00043567000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.002998199,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.001102039,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.004509099,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00137782,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00016182000000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.00299912,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00024807000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00110373,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00461351,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00032988,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.004635670000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00462741,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00114425,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00043638000000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029770300000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0003305,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00031821000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00032237000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.0003264,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00113535,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00285036,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0028631100000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004623680000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00109903,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0031776900000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00032411,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00032297,
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
          "id": "670b480323cb957ce94a48d719b8ec95e8a3e15e",
          "message": "fix: unwanted promotions in abstractrange (#1677)",
          "timestamp": "2025-09-17T19:07:47Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/670b480323cb957ce94a48d719b8ec95e8a3e15e"
        },
        "date": 1758166488109,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0027640620000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.002834215,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.0068642880000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.006667628,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.006256986,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.006593565,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.006250632000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.006878530000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.003002134,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.006976556,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.006424152000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.002926595,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.006283277,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.006926369000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.006892713000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.006415914,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.006691506000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.0027958270000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.006947857000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.006691673,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.003069885,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.006662285,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0020071,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000661261,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0007559890000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0071443,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0006784860000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029501220000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0019970020000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00722341,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0030136890000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.000354015,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.00298859,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029435570000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001086989,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007233908000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007207276,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007154596,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001114535,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0007195210000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.00032367,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0031074640000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.003105263,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000664801,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.010420036,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.002928972,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007123775000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00031832000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.001134737,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.002934453,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.003332816,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007222719000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.00305555,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0030500780000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007097975,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007120118000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.000311184,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0031280170000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000670725,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0006503990000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007114771000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0006777290000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007130952,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.003127176,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.0072779690000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002010444,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030678090000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000653007,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0029185030000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.00711759,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006618440000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001167541,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0024550170000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002064718,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0070958340000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029879990000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.0031179190000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000691727,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001091488,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007138152,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000659727,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0006512950000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0029385260000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.002939876,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0006570590000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.002004702,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001095781,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000305221,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0020108020000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010502326000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000662947,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000313138,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002955905,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.002985886,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007230926,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0006883530000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004745551000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00117253,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00300263,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.00218447,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00033360000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00033629,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00289915,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.00300393,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00475926,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00033661000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00016552,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00475095,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00117157,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00301906,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00033484000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00033613,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00044488000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0029921600000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003001259,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.004771500000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00111234,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00299928,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.0007893900000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00246953,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00117095,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.0001644,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00033065000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00472845,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00016663000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0011814500000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00301111,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00044431,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00033088000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00015879000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00033349000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00477299,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00300394,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00033743,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0047531000000000006,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0047541300000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00474663,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0028714500000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00478295,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004757470000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.0013455000000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00045458,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.00045470000000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0029963900000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.00111345,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.00463068,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.001380411,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00014961,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.0030175700000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00026558,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00110917,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.004758330000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00033344,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.004773540000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0047601100000000006,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0011653500000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00044451,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00300108,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00033571,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00033867,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00033584,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00033564,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00113881,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00288199,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0028984,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00476311,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.0011101000000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0031764500000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00033436,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00033649,
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
          "id": "6da23fae188be0e6ff5d5d50ddcc4e66e6c9b076",
          "message": "Format Julia code (#1679)\n\nCo-authored-by: enzyme-ci-bot[bot] <78882869+enzyme-ci-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-09-18T14:49:02Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/6da23fae188be0e6ff5d5d50ddcc4e66e6c9b076"
        },
        "date": 1758253073446,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.001912438,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.0016747320000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.0041877550000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.004121792,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004215789,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.0041999310000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.0041512630000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.004020043,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.0017805260000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.004474179,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0041100450000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.001804044,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.00399288,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004210701,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.004413963,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004285135000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004253952,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.0016679230000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.0041661020000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.004215082,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.0019242130000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004337225,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002022106,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0006569950000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0007804110000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0071207860000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000680805,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.002974837,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002009098,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007254129000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0030004550000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.000357351,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0029909420000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029745590000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001083084,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007246618000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007205137,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007149372,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001072026,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006515900000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000313212,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003158005,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0031170200000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000670076,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.010359625,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0029202200000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007130682,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.000307005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0011397690000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0029518540000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0032953910000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007224254,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.003063517,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003051957,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007125037000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007136679000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.000328077,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0031046570000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000661089,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000658099,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007166591000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000671685,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007141757,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0031286150000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007248441,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002015825,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003065238,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000675157,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002933105,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007110954,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006627460000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.0011696760000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002444625,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.0020665230000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007130463,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.002982602,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.0031598190000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000687133,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001093688,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007158775,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000672493,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.000661717,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0029294800000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029535720000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000656584,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.002012439,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001102383,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00030645,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0020113640000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010393642,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.00069182,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000321338,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029237340000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.002985658,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007243124,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0006825010000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00473024,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00118186,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00300057,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.00219863,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00033733,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00033794,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00287921,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0030057400000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.004776610000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00033814,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00020963,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00476398,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.001185089,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0029922200000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00033446,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00033782,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00044207000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0029969600000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00300029,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.004757760000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0011171590000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0029899400000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00077004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00249963,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.0011741400000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00019918,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00033763,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00475444,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00020638,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0011676800000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00297999,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00044050000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00033821000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00019563000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00034891,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.004782379000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00299355,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00033909,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004765221,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004772061,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.004749119,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00288834,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00476412,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004761130000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.0013614100000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00044363000000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.00043886,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.003007421,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0011138,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.004643351,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.0013754300000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00020751000000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.00298677,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00029434000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00111137,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00476831,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00033869,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.00477328,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00477291,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.001170031,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00042902,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00301678,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00033754,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000316211,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00033600000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00033637,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0011373000000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0028714500000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002867,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0047555,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00110814,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0031803900000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0003373,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00033596,
            "unit": "s"
          }
        ]
      }
    ]
  }
}