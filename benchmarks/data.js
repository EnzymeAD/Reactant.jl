window.BENCHMARK_DATA = {
  "lastUpdate": 1761968612733,
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
      },
      {
        "commit": {
          "author": {
            "name": "William S. Moses",
            "username": "wsmoses",
            "email": "gh@wsmoses.com"
          },
          "committer": {
            "name": "William S. Moses",
            "username": "wsmoses",
            "email": "gh@wsmoses.com"
          },
          "id": "0176cd7e2870f539f06d81e888bc70230470e13f",
          "message": "proto patching",
          "timestamp": "2025-09-21T02:34:44Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/0176cd7e2870f539f06d81e888bc70230470e13f"
        },
        "date": 1758428906239,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.001916888,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.0019325890000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.004434989,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.004295449000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004293865,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.0042236090000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.004218212,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.004192306000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.0017826500000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.004492983,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004238115000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.002010502,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.004136832,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0041825230000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.004396764,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004324011,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0043524720000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.001570975,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.004247903,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.004344436,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.001984299,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004285357,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002088894,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000657132,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0007915390000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007174543,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000677464,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029832680000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0020830790000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007283512000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.003010879,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00033058500000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0030201340000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.003003167,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001077354,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007256269,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007240052,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007139085000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001096469,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.000661557,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000311647,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0031885060000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0031858060000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0006752290000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012259209,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0029560410000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007175393,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.000315213,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.001133567,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0029685030000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0033643970000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007236154000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.003079848,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0031531790000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0071170370000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007137954,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.000309807,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003138891,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.00067805,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000659528,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007126348,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0006908890000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007217109,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0031619160000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007328502000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002066835,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0031004170000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.0006836510000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002985596,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007133634000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006706910000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001185674,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0025082670000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.0021648780000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0071509170000000006,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003003706,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002566782,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0006809310000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001085004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007161817000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0006567470000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0006635660000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002981033,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.002980051,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000676653,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.0020863590000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001097716,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000335579,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0020930230000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010491754,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000684012,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000349351,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002979423,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003057745,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007256498,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0007048950000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00462423,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00128915,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00300758,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.002744849,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00030543000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00031001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0028618100000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.00299188,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00462832,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00031262,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00017614,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00462301,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00127369,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00299751,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00030315,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00030968,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00061346,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.002995371,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00298064,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.004624110000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00111308,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0029934500000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00100147,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00246429,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00129239,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00016124000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00030415,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.004610639000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00017069,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0012849900000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00299751,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0006135400000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00030621,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00016978000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00032033,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0046082300000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0029966000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00031792,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00461826,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0045964890000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0046172,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0028563100000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004629230000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0046332000000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00120676,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0006258500000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0006125000000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00300101,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0010980900000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.00449655,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00137258,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00016568,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.0030012800000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00025229,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0010971700000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.004631719,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00031272,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.00465192,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0046129100000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0012944500000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00060977,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029954400000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00030791,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00031507000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00031135,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00030416000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00113202,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0028653700000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00286842,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0046187400000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.0010957100000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.00316341,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00030048,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00031101,
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
          "id": "4b94dd80a8585fd5f5a4b625f0e0b023e772db29",
          "message": "Update ENZYMEXLA_COMMIT hash in WORKSPACE",
          "timestamp": "2025-09-22T02:48:39Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/4b94dd80a8585fd5f5a4b625f0e0b023e772db29"
        },
        "date": 1758514636702,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.001597494,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.001508295,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.0041198730000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.004189241000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004070331,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.0040009500000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.004081894,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.004028823,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.00152635,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.004205195,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004133433000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.001780829,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.0038993630000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00402195,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.004118575,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004064446,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0040917570000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.00149747,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.004058374,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.004037293,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.0015570450000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004123008,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0020707060000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000692234,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0008321530000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007184586000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000715722,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.00299176,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002074369,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007295944,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0030516280000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.000352518,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0030792190000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.003049459,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001108753,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007275817,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007249982,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007152335,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001092467,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006698370000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000317244,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00318562,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0031694640000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000691118,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012280422000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.002997717,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007198253000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00032197000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.00115095,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.00301765,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0034544040000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007285791000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0031383970000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003166407,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007130920000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007193238,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.000330441,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0031510540000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000675186,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000689576,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0071484800000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0006937180000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007181154,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.003189237,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007348913,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002082342,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0031427150000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000683938,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002971219,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.00714648,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000675943,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001201108,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002547934,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002126027,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007157498000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003063159,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.0025707300000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0006891680000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0010998210000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007204847,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000665099,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.000660985,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0030618610000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029741230000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0006715260000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.00207682,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001098075,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000317268,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.00207851,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010513731,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0006782050000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.00033000900000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003015565,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030369000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007286058000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0006853040000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00463447,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00126807,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00300057,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.00277755,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0003278,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00033033000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00286345,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0029884200000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00462814,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00031874000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00018305,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.004647790000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00130207,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00296613,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00033213000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00033444000000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00061928,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0029771000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.002981951,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00463938,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0011062300000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0029846,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00099952,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00248385,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.0012975,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00017559000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00033064000000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00462402,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00018386000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00130082,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00300629,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0006105,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00033391000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00017014000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00034001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.004632001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00298318,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00033275000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00463366,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00463516,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.004631720000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0028683700000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00462599,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00463789,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.0012368400000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00062159,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0006137600000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00298721,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0011084900000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.00449774,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00137696,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00016825000000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.00299145,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00027442,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0010975400000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00462557,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00033691,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.004631290000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00462565,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00128893,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.000611,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029841800000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00033584,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00032001100000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00032189,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00033022000000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0011361300000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00286467,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0028880100000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00462494,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.0011028090000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.00315092,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00032784000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00033402000000000003,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "William Moses",
            "username": "wsmoses",
            "email": "wmoses@google.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "b39a1fcf45fb71cf63aa3a3657d3607ab9b81e3a",
          "message": "Bump version from 0.2.164 to 0.2.165",
          "timestamp": "2025-09-22T23:46:34Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/b39a1fcf45fb71cf63aa3a3657d3607ab9b81e3a"
        },
        "date": 1758598346525,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0012314250000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.0013988590000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.0031042300000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.0032064470000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0032746610000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.003211999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.003230431,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.003196397,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.001445833,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.003238738,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0031780700000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.0015866250000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.002945861,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.003190235,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.003235322,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003160636,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0032158760000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.0013816990000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.0031022040000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.003247835,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.0014489700000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003132812,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002074279,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000669327,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.000807588,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007155723,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0007139280000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.002977113,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002073543,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007302706000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.00300874,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00036472500000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0030262080000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.002972582,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001081441,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0072441250000000006,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0072261980000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007142184,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001107618,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006696800000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000313226,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0031764890000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.003202068,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0006860340000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012149124,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.00300876,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007168047,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00031822100000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0011305240000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.003010773,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.003373024,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007233693,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.00309208,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0031838770000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007123079,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007138164000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00032424600000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0031313020000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000708507,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000670026,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007131262,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0006976850000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007128969000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0031221870000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007285312,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0020806320000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003113324,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000694197,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0029654720000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007125631,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000659374,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001173835,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0024993890000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002109721,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007160196000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003055518,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002544157,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000695039,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0010839460000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007129413,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0007169510000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.000655514,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002998025,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029536980000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000664279,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.0020990590000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.0010937890000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000316157,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.00206522,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010443821,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000743514,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.00032548200000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029939040000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030197640000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007255219,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000674785,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0046454700000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00129855,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0030290800000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0027870900000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00034612,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00034512000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0028706,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0030045700000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00462869,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00033072,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00015147,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.004642240000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00129721,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0029911,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0003465,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00034612,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00062765,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00299775,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0029857300000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00463835,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0011029,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00301159,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00101604,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.0025075600000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.0012931400000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00015562,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00033021,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00463749,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00016377,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00131598,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030083600000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0006344100000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00033204,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00016731000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00033902000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00463892,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00299765,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00033096000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00463984,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00464449,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.004646480000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00288153,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0046429,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004638321,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.0012128500000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0006365100000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0006322000000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0030121500000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.00110753,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.0045174690000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00139105,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00015586000000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.00300904,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00027009,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00110486,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00464344,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00034008000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.0046367000000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00463868,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00131162,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00063407,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030054300000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00034652,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00032831,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00035493000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00034567,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00113096,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0028710100000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0028641,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00462506,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.0011061,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.00316824,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00035139000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00032933,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "William Moses",
            "username": "wsmoses",
            "email": "wmoses@google.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2131b2788597c861cd85b7c364a8dd7f5bbcf68f",
          "message": "Bump version to 0.2.166 and update Reactant_jll",
          "timestamp": "2025-09-24T03:15:35Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/2131b2788597c861cd85b7c364a8dd7f5bbcf68f"
        },
        "date": 1758684844432,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0024067840000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.002254101,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.005826802000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.005924193,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.005839057000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.005868277000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.00568523,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.005890702,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.002402238,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.006035788,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005928088000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.0027047720000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.005446732,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.005861321,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.0060816590000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005698693,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.006128887,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.00244653,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.0060840880000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.006018556,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.0026089150000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.005988157,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.00207267,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000659719,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.000781538,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007126589,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0006768330000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.002952203,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002088262,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007271700000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.003008483,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00031795,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0030012750000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029658870000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001096432,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007244951,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007269816,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007139463,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00109638,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006655410000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.00031572100000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003225472,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0031815460000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0006713520000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012320307,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0029470190000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0071298460000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.000312852,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0011316800000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0029653120000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.003401124,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007224094,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0030645620000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003162512,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007115695000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0071315860000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00030670800000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003136779,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0006719590000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0006703060000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007114193,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000683454,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007128227000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0031011740000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.0072945530000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002086091,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030906840000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.0006604100000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002950761,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007119805000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000666676,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001170003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0025442660000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002106246,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007130592000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003008989,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.0026219620000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000684178,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0010816130000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007133554,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000675371,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0006535110000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002966312,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029365600000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0006594950000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.0020791380000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.0010892480000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00031698800000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002082716,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010472342,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000728067,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.00031147300000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002953442,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003013833,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007236543000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000674917,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00461879,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00130101,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0030014100000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0027932,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00034408,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00032938,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00286118,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.00301053,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0046343600000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00034229000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00015838,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.004628500000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00129856,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00298065,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0003303,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00033314,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0006215400000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00296983,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0029898100000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00463141,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0011204300000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0029917700000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00101515,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00246533,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00128643,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.0002035,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00036082000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00462008,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00019098900000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00128951,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00299278,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00062405,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00036066,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00019264,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00037036,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0046303,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0030121310000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00034628,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00461928,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0046304300000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.004631290000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0028658200000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00462901,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004632819000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00122663,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00062295,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.00060841,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00301759,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.00111681,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.00453087,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00139206,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00018154,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.003002799,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00028878000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0011168900000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00463045,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00036080000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.00462816,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00464774,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00129289,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00061041,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030071300000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00033028,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00033678,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00032839,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00033048,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00114841,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0028636900000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0028570400000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0046275100000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00111677,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0031743400000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00033025000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0003396,
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
          "id": "79752b735b14b23d56687a1fd82531ca51dcffc6",
          "message": "feat: enable new auto-batching passes (#1690)\n\n* feat: enable new auto-batching passes\n\n* fix: move passes\n\n* fix: bad rebase\n\n* test: run acos/acosh with fp32\n\n* test: mark acos and acosh tests broken on tpu",
          "timestamp": "2025-09-25T03:10:09Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/79752b735b14b23d56687a1fd82531ca51dcffc6"
        },
        "date": 1758772147253,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0014162390000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.0012120120000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.003176468,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.003218289,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0031814020000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.00315912,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.003174141,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.0031593560000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.0012134720000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.0032922900000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003238631,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.001486516,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.00288375,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.003092319,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.0030873660000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0032189180000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0031673960000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.001381833,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.0032286130000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.0030958540000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.001354802,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003135503,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002070143,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000681409,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.000791247,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007130160000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0006973140000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.002964468,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002061786,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007244356,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.002999662,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00036927300000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.003012046,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029737820000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.0010915570000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0072519260000000006,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007235821000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007147835000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001082938,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006817790000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000347715,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003170758,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0031883050000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0006735500000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012473857000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.002955749,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007152698000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.000311909,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0011415750000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.002966793,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0034534300000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007239906000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.003076446,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0031677880000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.00714676,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007130717000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00032767,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0031229200000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000691326,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000682772,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007127338,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000701468,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007162514,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0031021810000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007294376000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0020601670000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003089474,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000686222,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002965595,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0071694060000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006754490000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.00116832,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0025328240000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002113027,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007150012000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029983590000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.0025689930000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0006975190000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001095267,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007128990000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000676729,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.000685622,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002969985,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.002953474,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0006890220000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.002070326,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.00108401,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00031917500000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002068687,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010457479,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0006868250000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.00035576600000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002953807,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0029907970000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007258467,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0006918940000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0046355300000000006,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.001294919,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.003001599,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.00276135,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00032799,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00032181,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.002868349,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.003010889,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.004655291000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0003215,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00016109000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0046481000000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00129699,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.002949049,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00033025000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00033248,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0006093,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00297677,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0029900810000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.004631140000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0011000600000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00302203,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.0009891000000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.002455009,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.0012673200000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00016439000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00032487,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.0046439,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00016783,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0012526,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00299957,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00060243,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00033634,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00016088000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00033021,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00463183,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0029688400000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00033032,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00464742,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004641119,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00464638,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.002871979,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00463746,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0046311,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00119761,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00059285,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.00058849,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00299103,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.001111,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.004528470000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00137512,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00016082000000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.00300067,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00026012,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00109694,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00466701,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0003282,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.0046391,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.004627079,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00125796,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00060422,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00300349,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00032444,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00032162,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00032736,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00032713,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0011346000000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0028586500000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0028758800000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0046479600000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.0010986400000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.00317013,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00032249000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00032595000000000004,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "William Moses",
            "username": "wsmoses",
            "email": "wmoses@google.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "2d0e0e306158059a6b1e6503c0fc7153b77789f5",
          "message": "Use Enzyme.ignore_derivatives, now that landed (#1707)\n\n* Use Enzyme.ignore_derivatives, now that landed\n\n* move\n\n* fix\n\n* Apply suggestion from @github-actions[bot]\n\nCo-authored-by: github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>\n\n* Update ReactantABI to use EnzymeCore directly\n\n* Change ignore_derivatives reference to EnzymeCore\n\n* fix: docs\n\n---------\n\nCo-authored-by: Avik Pal <avikpal@mit.edu>\nCo-authored-by: github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>",
          "timestamp": "2025-09-27T03:33:37Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/2d0e0e306158059a6b1e6503c0fc7153b77789f5"
        },
        "date": 1758954101436,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.002333824,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.0024646570000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.005851518,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.005744168,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00584645,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.005970990000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.005763497,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.005751776,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.001989602,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.005977978,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005798264,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.0023129770000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.00540511,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.005606812,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.005950676,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005852118000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0057234930000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.002216528,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.0058958510000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.0056800390000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.002619086,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.005811745,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.00205274,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000666681,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.000762898,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007140102000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.00068922,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.002956025,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002086029,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007256696000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0030004890000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.000353448,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.002982953,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.002941069,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001079076,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007260336,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007200178000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0071715270000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001082031,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006877070000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000313609,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003031025,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0030076150000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000663109,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012531197,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0029613630000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007156228000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00030943100000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.001128237,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.002966536,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0033995550000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007219748000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.003082965,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0031736430000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007160014,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007192657000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.000310559,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0031383690000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0007027600000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0006755110000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007133604000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000692622,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007166425000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0031079740000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007289716000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002062013,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00308581,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.0006769240000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0029300380000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0071556580000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006768760000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001177242,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002500468,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002105229,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007139545000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.002990716,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002564438,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0006905030000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001076218,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007178604000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0006938710000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0006731,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002975278,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.002958931,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000667563,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.0020656610000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.00108094,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00030484,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002061082,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010384927,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000697784,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.00031803,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029525930000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003025728,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0072179570000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000680878,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004753511,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00131812,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.002986589,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0027600700000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00031293,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00030559,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0028602500000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.00299247,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0047504,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00032038000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00013723,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0047601100000000006,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0013204,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00298451,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00029294,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00031408,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0005855200000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0029936800000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00297393,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00476831,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00110106,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00298165,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00097194,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.002441389,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00130422,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00013803,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00030228000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.004773071,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00013897,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.001311529,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.002980459,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00058808,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00030489000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00013801000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00031116,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00477972,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0029771000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00031967000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0047913,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004755560000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00474337,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00285856,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00475581,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0047450800000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.0012379490000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0006199900000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0005869800000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0029911300000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0011114,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.004641671,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00137232,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00013761,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.002984959,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00024262,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0010956100000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00476018,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00030704,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.00477421,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0047642000000000006,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00130769,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0005898300000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00298574,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00030522000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00032623,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00031790000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00030461,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00113138,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00286177,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00287001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00478469,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.0010938500000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0031706,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00030871,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00031327000000000004,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "William Moses",
            "username": "wsmoses",
            "email": "wmoses@google.com"
          },
          "committer": {
            "name": "William Moses",
            "username": "wsmoses",
            "email": "gh@wsmoses.com"
          },
          "id": "c254e64bbd20ee646f02b23f8b86a53a485ce6ca",
          "message": "Bump version to 0.2.168 and update Reactant_jll",
          "timestamp": "2025-09-27T16:09:51Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/c254e64bbd20ee646f02b23f8b86a53a485ce6ca"
        },
        "date": 1759032292331,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0020359930000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.0017621050000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.004368765,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.004300539,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004384716,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.004402299,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.004172497,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.004390467,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.001905177,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.004376851,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00433476,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.0018597560000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.004476255,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004347374,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.004415989,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004509362,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004418109,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.001770106,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.004333931,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.004349436,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.0016250840000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004386323,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002079121,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000697536,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0007786070000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007145968000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0007158140000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.002982362,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002074134,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007284074000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.003006714,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.000361742,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0030229560000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.00297392,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001114397,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007206846,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007236457000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007139071,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0011121570000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.000691979,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000319145,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0032165260000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0032338590000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000695224,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012248478,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0029758000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007160114,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.000329832,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.001171411,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.002977996,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.003445379,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007186142,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.003079899,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003197454,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007121329,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007151018,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00033114900000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003137491,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000699532,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000707136,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007114935,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000724686,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007163852,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0031268750000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.0072761530000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0020763690000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003088986,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000698719,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002967827,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007143021,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000696251,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001207437,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002533534,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.0021109880000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0071292230000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030196840000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002570976,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000717504,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0011000410000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0071381510000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000705491,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0006982480000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0029804270000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029556120000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000696565,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.0020911880000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001114713,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000333771,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0020835280000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010091026000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0007193340000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000326654,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002954749,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030180610000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.00722931,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000717244,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0047519400000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00130918,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00302162,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0027915500000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00035083,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00034931000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00289302,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0030309900000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00476285,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00033923000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00019926,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00477146,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00129582,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030234600000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00034940000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00034645000000000005,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00063195,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00304404,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0030064000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.004771269000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00112125,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00302737,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00101011,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00247618,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.0013037300000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00018912,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00035386,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.0047799800000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00018189,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00130425,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030350000000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0006314900000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0003532,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00019539,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00035126000000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.004790941,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00301752,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00036714,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00477986,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00479319,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.004776740000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0029119090000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0047566100000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00476665,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.0012354500000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0006420600000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.00063944,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0030197600000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.00111961,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.004646039,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00141398,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.0001976,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.003026141,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00028940000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00111716,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0047615900000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00034996,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.00476429,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0047718100000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0013049700000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0006338800000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00302811,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00035025000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00033898,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00034778,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00034548,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00114412,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0028996200000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029048100000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00475503,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.0011136800000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.00319145,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00034393,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00034444,
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
          "id": "74ac7035b34243028baaedceeeb176d6a01adf79",
          "message": "Format Julia code (#1715)\n\nCo-authored-by: enzyme-ci-bot[bot] <78882869+enzyme-ci-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-09-28T22:10:45Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/74ac7035b34243028baaedceeeb176d6a01adf79"
        },
        "date": 1759117011061,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.001357219,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.001152171,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.0031241750000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.0030392960000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003043787,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.0032289370000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.0031070940000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.003129882,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.0013045040000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.0031343630000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030998420000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.0015450260000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.003195837,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0031021760000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.0032662520000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0030429790000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0031662920000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.001233947,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.0030558100000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.0032073360000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.0011674130000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030423150000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002080158,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000675682,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0007651120000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007147876,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0007091440000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029602960000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002082554,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007302449000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.00300395,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00036999200000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0030123190000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029696510000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.0010933120000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007225775,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007223906,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007121702000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001097815,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006896000000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000331164,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0032356570000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.003220592,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000705292,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012240099,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.002954987,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007145452,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.000315863,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0011593150000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.002976752,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0034952760000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.00722038,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0030855590000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0032044730000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007106524,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007173550000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00032725,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003135039,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0006897180000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000705374,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007119497000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000696774,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007121079000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0031297820000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007289362000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002082941,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003096502,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.0006994950000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0029579610000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007115919,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000698722,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001189806,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0025506720000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002114032,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007196702,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030126510000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002551659,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0007070360000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001097054,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007138074,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0007013060000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.000691076,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002976223,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.002942608,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000687959,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.002069668,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001092281,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00031534900000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002073697,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010194700000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000710648,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000331492,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029574690000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.002990197,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007207333000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000698211,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0047326,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00130886,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00301001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.00275699,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0003389,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00034013,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0028829800000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0030003100000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00475392,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00033584,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00020332000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0047586700000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0013030700000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00300036,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00034283,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00033607,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00060386,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00300913,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00299395,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00473592,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00110876,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0030256000000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00099811,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00245255,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.0013047500000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00020141,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.0003368,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.0047269090000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00020179,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0013181900000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.002998859,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.000612339,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00033369000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00020172,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00035169000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00474398,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0030396700000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00033013000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00472775,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00475202,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.004756880000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0028689,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0047488800000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00475904,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00124944,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00062549,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0006262,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0030269900000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.00111347,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.00463367,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.0013847100000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00020139000000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.002999601,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00029620000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00111516,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.004728260000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0003422,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.00474393,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00474102,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00129213,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00060342,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030108400000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00033483000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00033471,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.000341,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00034221,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00113962,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00287161,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0028998400000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004769450000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00110785,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0032012200000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00033199,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00033652000000000004,
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
          "id": "85bec8393a30f4097788982263e81f221dcd3d87",
          "message": "fix: add additional check in ignore_derivatives (#1717)",
          "timestamp": "2025-09-29T16:44:36Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/85bec8393a30f4097788982263e81f221dcd3d87"
        },
        "date": 1759205112619,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.002560947,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.002281209,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.005850255,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.005763394000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.005882330000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.005844299000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.00584183,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.005867077,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.0021767170000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.006308523000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0060247040000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.0028832130000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.00549368,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.006047259,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.006027192000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.006017944000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00580793,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.002674328,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.005953793000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.0060548040000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.002684127,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.005828664000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0020522350000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000712679,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.000847106,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007137893,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000698074,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029783400000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00205432,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007244868000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0029951820000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.000347948,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0029701650000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.002944233,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.0011006330000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007202563,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007182646000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0071518120000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001115256,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.000673489,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000333918,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003166705,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.003165671,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000655589,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012189732,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0029577260000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007108733000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00032877700000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.001128591,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0029670620000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.003417705,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.00719339,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0030851570000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0031460660000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0071043880000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007118808000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00033255400000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00313473,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000689881,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0007223970000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007076926000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000717477,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007119236,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0030932250000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007223787000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002061786,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030785580000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000668335,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002926193,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007116529000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0007113,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001172834,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002536705,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.0021107170000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007117115,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029998710000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002574405,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0007282020000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001085727,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007139011000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0007104020000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0007075810000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0029748910000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029204870000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0007083370000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.002083857,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.0010789970000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000319713,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002056233,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.01048516,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000729564,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000310142,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002934739,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0029988180000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007195765000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000753532,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00474411,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0013243,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0030304800000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.00277331,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00033102,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00033638,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0029040800000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.00302725,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0047444390000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00033328000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00015816000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00475186,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0013262,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00298467,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00033904000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00033597,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00062887,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00298801,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00299612,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.004761110000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.001114589,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0030393,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00101089,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00247145,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00129727,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00015816000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00034437000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00476694,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.000177839,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0013020800000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030244200000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00063407,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00033387,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00016748000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00034209,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.004770320000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0030133900000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0003324,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00478295,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00477903,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0047682,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00289533,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00477022,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00475218,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00123217,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0006488,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.00065227,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0030053500000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0011190800000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.004666469,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.0013812100000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00016007,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.00302994,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00028205,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00111432,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.004757391000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00033897,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.004765640000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0047928300000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00131729,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00063028,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030041300000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00033521,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00033107000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00033728,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00033437,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00113839,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00286301,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029000600000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004747530000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00111768,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0031924600000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00033513,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00032892,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Acake",
            "username": "sbrantq",
            "email": "scharfrichterq@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "1ec34529bc42b3521aabb3baba15bd8ba9f1ecdb",
          "message": "ProbProg: JLL changes for trace/symbol/constraint types (#1719)",
          "timestamp": "2025-09-30T19:47:52Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/1ec34529bc42b3521aabb3baba15bd8ba9f1ecdb"
        },
        "date": 1759290579548,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0019800910000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.0016334680000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.004171298,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.004168444,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004139061,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.004247576,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.004211137,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.0041465600000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.001714625,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.00432232,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004127978,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.002037608,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.0038860110000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004122156,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.0041326,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003997627,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0040688320000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.0015370030000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.004187323000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.004058855,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.001576113,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004169548,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002062613,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000668665,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.000839808,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007116296,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000695056,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029453920000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0020554830000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007271391,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.002967742,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00036456300000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.00298413,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.002939992,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001086031,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007215986000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0072274620000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007133066,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0010788010000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006600800000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000321521,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003194432,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0032344920000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000667459,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012006307,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.002944084,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0071241510000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00031177500000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.001130843,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0029591670000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0033720810000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007217544,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.003062649,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003163901,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0071048890000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007156314,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00031945700000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0031271420000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0006645150000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000665276,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007119023,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000682596,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007130363000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0030977310000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007250378,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002057667,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003056137,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.0006587390000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002923689,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0071016230000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000670047,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001166416,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0025191470000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002119855,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007166675000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003000019,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002563393,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000689128,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0010943980000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0071528270000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000678564,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0006567550000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0029498000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.002942838,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000661834,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.0020632100000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.0010770110000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000306823,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0020684370000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.01041565,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000701098,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000314779,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002937452,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0029955470000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007198520000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0006884910000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0047551600000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00130527,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0030057300000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0027675,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00033418,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000336721,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0028689500000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0029801700000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0047318,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00032205,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00017299,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.004739529,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00129608,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0029851200000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00033303,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00033299,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00062664,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0029979900000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0029786810000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00476293,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00109303,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00301546,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.0010053100000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00247239,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00129112,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00014018,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00031221,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.004754840000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00016078000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0013206300000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00301321,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00062635,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00031825,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00014515,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00031911,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00475083,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0030144,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00032300000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004749211000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004772230000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.004747229,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00285791,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0047319810000000006,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004737760000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00123058,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00064249,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.00062821,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00299304,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0010922010000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.0046305700000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00136713,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00015919,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.0030177000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00027543,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00108964,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0047536,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00031298000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.004748451,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00475051,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00130341,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0006271600000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030059600000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00033054000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00031648,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.0003322,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00033466,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00111223,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00287453,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00288148,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0047697500000000006,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00108646,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.00315789,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00033528000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00032017000000000005,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Acake",
            "username": "sbrantq",
            "email": "scharfrichterq@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "1ec34529bc42b3521aabb3baba15bd8ba9f1ecdb",
          "message": "ProbProg: JLL changes for trace/symbol/constraint types (#1719)",
          "timestamp": "2025-09-30T19:47:52Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/1ec34529bc42b3521aabb3baba15bd8ba9f1ecdb"
        },
        "date": 1759376858760,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0017131870000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.001807407,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.004180098,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.004245553,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004239583,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.004235513000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.004232373,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.004202548,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.001704168,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.004430215,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004378906,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.002024391,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.004181487,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004210826,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.004220423,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0042431100000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0041399850000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.0017712670000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.004184814,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.004242593,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.0015953550000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004257847,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002058826,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000716268,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.000830662,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007133824,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000714581,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.002948771,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002106592,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007239898000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0029875220000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.000362622,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.002959812,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029467990000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.0011024140000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007207023000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007205593000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007132221,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0010852350000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.000698741,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000374491,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003215573,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.003167842,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000709255,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012372316000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.002928033,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007132251,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00038123400000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.001129046,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0029238830000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.003395686,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007287428,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.003076465,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003168082,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007094129,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007121433000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00033909,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003144637,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000708746,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000703251,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007110756,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0007054800000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007129852000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0031473720000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007302449000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002062983,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003050206,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000676575,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002916655,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007098696000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000681686,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.0011824860000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002555387,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.0021115070000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007111504,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029671420000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002604,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000692001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0010927950000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007111489,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000678656,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.00066922,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002927985,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029269440000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000674599,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.00206387,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001085826,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000336116,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002087659,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010701932,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0007224870000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000337992,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002931795,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0029806900000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007221611,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000709276,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0047447900000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00131741,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0029983690000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0027787600000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00033611000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00033954000000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0028762500000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.00300657,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00473755,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00033685000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00014896,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0047415800000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00130251,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00301531,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00034404000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00033621000000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00062219,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0029878,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0029913500000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.004760960000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00110888,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0029945,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00099368,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.002466359,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00131069,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00016342,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00032923,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.0047428900000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00017323000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00130753,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0029837400000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00063185,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00033263000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00017355000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00034156000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00476264,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0030034700000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00033926000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00476487,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0047553000000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00472868,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.002871419,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00473707,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00474049,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00123864,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00062912,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0006259000000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.002984949,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.00112099,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.004648149,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.0013865400000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00017107,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.003001899,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00024752000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00110958,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00476372,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00034843,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.00473667,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00476006,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0013067900000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00062169,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030005500000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00034106,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00034276,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00034153000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00033508,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0011423100000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00284991,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00287387,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00475232,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.0011079800000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0032040790000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00034578,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00033646,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Acake",
            "username": "sbrantq",
            "email": "scharfrichterq@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "1ec34529bc42b3521aabb3baba15bd8ba9f1ecdb",
          "message": "ProbProg: JLL changes for trace/symbol/constraint types (#1719)",
          "timestamp": "2025-09-30T19:47:52Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/1ec34529bc42b3521aabb3baba15bd8ba9f1ecdb"
        },
        "date": 1759462275820,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.001682698,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.001708964,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.0039849560000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.004062872,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004044663,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.004224314,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.004167297,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.004153812000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.0015627640000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.0041607960000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004257521,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.0019271870000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.0038911160000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004224681,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.004159576,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0041277860000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004118491,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.00155034,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.004071219,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.004113205,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.0018085610000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0040266880000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002062179,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0006861430000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0008373920000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007121715000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0007090750000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029397100000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002067068,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007217823000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0029893520000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00035451000000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.003005896,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029495880000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001115962,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007204757000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00718823,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007140878000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001120502,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006715730000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.00032834000000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0032106060000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.003183298,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000683853,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012302055000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.002930357,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007126493,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.000327288,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.001140567,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0029439960000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.003379112,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007198467,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.003059247,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003156808,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007121219000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00712179,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00031844100000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003142367,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0006765580000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000662713,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007096099000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0007025740000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007105297000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0031000420000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007271410000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002053695,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030925460000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000680072,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002953041,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007124547,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000670088,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.00116956,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.00252001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.0020999120000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007113674,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.002971282,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002577537,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000705068,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0011168080000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007139299000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000678628,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.000676816,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0029478290000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.002924335,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0006630830000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.0020666250000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001092279,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00032167700000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002075417,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010561592,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0006929670000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000320681,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029247130000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003005142,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007212560000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000700551,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00475453,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00133098,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00299655,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0028036600000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00031479,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00031512,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00287397,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0029921400000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00476064,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00031427,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00017264,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00475889,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0013220600000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00298351,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00031192000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00030971000000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0006389400000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0029855700000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00299862,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0047512000000000006,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00111161,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00299845,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00099851,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.0024857800000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00129184,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00018068,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.000331289,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00477534,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00016690000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0013224500000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00299818,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00063685,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00033321,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00017385,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00033853,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.004751811,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0029882800000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00032076,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0047414200000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0047653090000000006,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00475175,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0028638300000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00477423,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00477273,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.0012303190000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0006619400000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0006206800000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00299634,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.00112005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.00465215,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.0013763500000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00016930000000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.0029876100000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00028024,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0011176200000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0047463800000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00033699,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.004774710000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.004755581,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00132889,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0006295900000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00299979,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00031996000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00031448,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00030972000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00031643000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00114587,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0028828400000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0028702190000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00477332,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.0011152500000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.003182919,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00030434,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00031087000000000004,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Acake",
            "username": "sbrantq",
            "email": "scharfrichterq@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "1ec34529bc42b3521aabb3baba15bd8ba9f1ecdb",
          "message": "ProbProg: JLL changes for trace/symbol/constraint types (#1719)",
          "timestamp": "2025-09-30T19:47:52Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/1ec34529bc42b3521aabb3baba15bd8ba9f1ecdb"
        },
        "date": 1759551569137,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.001236846,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.001214711,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.0032372950000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.0033289620000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003235962,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.0034270040000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.0032215100000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.00327382,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.001172174,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.003307017,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0032252630000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.001570517,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.0030855730000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0033308390000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.0033142600000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0032728270000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0032482300000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.0011794490000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.0032412670000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.0032724250000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.0011708760000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003290433,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002068996,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.00067309,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.000837299,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007155880000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000688187,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029989110000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0020728630000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007306415,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.003007085,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00037894400000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.003006344,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.002995348,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001097191,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007255037000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0073013520000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007172810000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001090758,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.000688976,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000327828,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0032329010000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0032362800000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000685032,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012349288,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0029770900000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007182095,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00038208100000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0011454780000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.002988199,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.003444452,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.00728932,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.003107438,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003168923,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007134233,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007183204,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00037715400000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003138389,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0006743900000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000668031,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007131923,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00075294,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007180536,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.003199129,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007334915000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002077354,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003083839,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000664498,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0029528880000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007129843,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006842650000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001175941,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002583012,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.00212675,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007142346000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030324930000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002609165,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000701935,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001089734,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007140484000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000665471,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.00065998,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002976405,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029706330000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000673923,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.0020918950000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001087541,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000360403,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002074182,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010643391,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000685674,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000316136,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002965791,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003018068,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007343103,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0006804490000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00477335,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00131909,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0030047700000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0027973,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00030516,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00030538,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0028908500000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0030054400000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00476561,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00030535,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00015094000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.004750320000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00130185,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030074100000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0003035,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00030299,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00065103,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0030083,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0029880400000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00474911,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00110202,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00300364,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.0010240400000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.0025128390000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00133361,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00014938,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00030841,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00475496,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.0001598,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0013280100000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030021690000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.000645349,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.000336149,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00014862,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00030805000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0047597690000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0030087900000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0003031,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00474759,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00476371,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.004756890000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00291056,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0047406390000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00475636,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.0012346500000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00064141,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0006174300000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0030004190000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0011088,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.0046523300000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00139145,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00015018,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.00300905,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00026188000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0011122,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00477927,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00030654,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.0047636200000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00476017,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00131546,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00064702,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029939800000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00030674,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00030602,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.0003168,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00030559,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0011306900000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00288468,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00290186,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00478607,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00110566,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0031757300000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00030553000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00030786,
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
          "id": "4d7dc1cc122becd4177215f4a9b6280e41be4a42",
          "message": "Format Julia code (#1728)\n\nCo-authored-by: enzyme-ci-bot[bot] <78882869+enzyme-ci-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-04T18:01:01Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/4d7dc1cc122becd4177215f4a9b6280e41be4a42"
        },
        "date": 1759636481664,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0024847750000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.002312546,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.0063834830000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.006344837000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.006358654,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.0062398900000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.006074293000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.0062987270000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.002353795,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.006037757,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.006140687000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.002811894,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.006111712,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.006034521,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.006187574,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00624668,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.006261152000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.002186832,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.006149859000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.006408339000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.0024025540000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.006325255,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002063381,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0007108080000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0008449400000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007124173,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000699066,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029698100000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002070036,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007292608000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0030076290000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00032158400000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.002991437,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029418340000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001085133,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007247141,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007270892,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007142566,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0010829000000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.000675901,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000337307,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0031910000000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0032125820000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0006865,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012239432000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.002959595,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007138438,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00032152,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0011338980000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.002960377,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0034131430000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007284052,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0030917650000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0031891140000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007101172,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007151831000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.000336063,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0031428880000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000695926,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000682214,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007113221,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0006995790000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.0071444690000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.003125328,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007312272000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002094921,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003080775,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.0006898480000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0029600250000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007112224,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006778730000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.0011728930000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002551405,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002111497,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007119492000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030020040000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.00256272,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0006843740000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0010851670000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0071359430000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000682278,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.000691711,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0029614240000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029387330000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000676478,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.0020963080000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.0010953010000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000343321,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002089608,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010445258,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0007004400000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.00034360900000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002927069,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0029970780000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007268622000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0006944630000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004783449,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0013458600000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0030306,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.00281204,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00034031,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00034074,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0029157500000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.00304477,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00477449,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00033650000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00018333000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0047434,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00134764,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030291800000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00034087,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00033647,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00063352,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00302949,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00301934,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00480258,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00112154,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0030390290000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.0010206100000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00250297,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00134099,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00018705000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00034068,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00478543,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00018457,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0013474700000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00304913,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00060684,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00033996000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00018898000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00034568,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.004785640000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0030364800000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00034034,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0047932,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00476251,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00475621,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00290725,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00478938,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0047583600000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00127545,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0006142400000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.00061459,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00306748,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0011287,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.00469225,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00141006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00018817,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.0030371300000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00028998,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0011169300000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00476179,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00034043,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.004776919,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.004782451,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00133601,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00060945,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003032209,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00033882000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00034199,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00033944000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00034288,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0011547500000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00290865,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029042300000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00476486,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00113171,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.00319601,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00033879,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0003375,
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
          "id": "2952cb097f072af092c74a50b4e81eefbef612ec",
          "message": "Regenerate MLIR Bindings (#1730)\n\nCo-authored-by: enzyme-ci-bot[bot] <78882869+enzyme-ci-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-06T00:45:50Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/2952cb097f072af092c74a50b4e81eefbef612ec"
        },
        "date": 1759722127121,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0020280510000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.002155414,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.005997463000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.005936586000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.006027243000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.005877806,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.005721406,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.006080013,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.0021003320000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.006145397,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0061423680000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.002439005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.005525267,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.006081016000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.005957475,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005730838,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.005780765,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.002184847,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.005739465,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.00569911,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.002164848,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.006034865,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002066226,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0006688950000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0008307690000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0071884570000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000676003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029740490000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002069796,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007322249,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.003009435,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.000309521,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.003019724,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029755980000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.0010883610000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.00727239,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0072865000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007152574,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001079377,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006620510000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000317117,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003235271,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.003232106,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0006777910000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012338422,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0029506050000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007155761,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.000323752,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0011566480000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0029695760000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.00339768,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.00729443,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.00308098,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003184833,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007164979,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007163751,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.000313705,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003145297,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000678441,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00067747,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0071348620000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0006771710000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007149178000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.003126565,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007331574,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002078597,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003095112,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000664142,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.00294536,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007150640000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006706150000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001181079,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0025388380000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002113106,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007133124,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029974050000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.00256922,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000681975,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0010841470000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0071713770000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0006777580000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.000667403,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0029708620000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.002951153,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000656344,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.0020646,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001079125,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00031210900000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002080179,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010510535000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000684674,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.00032336100000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002970822,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030033520000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007348389,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00069541,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004784340000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0013364800000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.003020379,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.00292719,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00033166,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00033251,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0029087,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.003019,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0047872900000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00033457,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00016964,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00479235,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0013288500000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030165300000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00033336000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00032679000000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00057615,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0029994500000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0030076990000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00482648,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00110344,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.003022021,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00102005,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00263612,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.0013365500000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00017514,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00034993000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.004801720000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00016494,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00132693,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030160300000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0005973500000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00034551000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00017236,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00035815,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0048074400000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00301715,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00032769,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0048015,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00482279,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.004793400000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00289347,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004800690000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00476266,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.0012605000000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0005971,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0005993,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0030162400000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0011022900000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.00469522,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00135862,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00016713,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.0030063900000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00027525,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0011023600000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.004783339,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00033791,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.00480023,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0048024700000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0013382600000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0005911400000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030387200000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00033517000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00032557,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00033627,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00032697,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00112483,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0028918100000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029061800000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004787560000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00109362,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.003189761,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00033844,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00032264,
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
          "id": "7cdd96b07f268027d98bdeb794ff8f46356e480e",
          "message": "feat: updates for new JLL (#1721)\n\n* Add new optimization patterns to Compiler.jl\n\n* fix\n\n* Update src/Compiler.jl\n\n* fix: use new commit\n\n* feat: use cuda cc functions directly [skip ci]\n\n* fix: drop attrs + disable (post) pad_dot_general for now [skip ci]\n\n* chore: bump JLL\n\n* docs: update\n\n* test: update buffer donation",
          "timestamp": "2025-10-06T21:01:23Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/7cdd96b07f268027d98bdeb794ff8f46356e480e"
        },
        "date": 1759808346834,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0025193760000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.0024232050000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.006108120000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.006144494,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.005944022,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.006132491,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.006037105,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.005932585000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.002101176,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.006179091,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005908975,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.002171928,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.005859745,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.006152183,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.0060126630000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005947814,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.006165575,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.002387353,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.006118517,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.005813934000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.00213216,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.005872891000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0020718900000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000666,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0007956510000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007129061000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0006767090000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029630950000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002059145,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007227886,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.002982488,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.000356894,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0029798890000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029377640000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001080115,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0072127580000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007268686000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007137325000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00107682,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.000656674,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000308578,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003181726,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.003203837,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000680998,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012611082000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0029467240000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0071558,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00032739600000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.001118284,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.002943345,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0034827520000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007233804000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0030876130000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0031470070000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007107059000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00712103,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00031304600000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003107453,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0006656800000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0006807800000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007127462,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0006875550000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007137711000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0030919740000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.0072702,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0020816520000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003069851,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000682644,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0029152460000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.00713579,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000661985,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001168717,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0025402280000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.0021443860000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0071302150000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029725560000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002534409,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.00068004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001068988,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007129348000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000682125,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0006619470000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002945221,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.002932155,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0006729240000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.002066037,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001077826,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00031697,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002079754,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010569072,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.00068592,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.00032149900000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002936372,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.002982759,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007235245,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0006765,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00479477,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00130966,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00302571,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0029321200000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00034551000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00036495,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0029065600000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.00303125,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00479314,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00034468000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00018996100000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.004813270000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0013026000000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030269200000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00036055,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00035702000000000005,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00060545,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0030268,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003023369,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.004772800000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0011066700000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0030251,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.0009941700000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00262741,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.0013351300000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.0001819,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00036056,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00479453,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00018177000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0013369500000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00303459,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0006017300000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00036063,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00018835000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.0003729,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.004793240000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0030344200000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00034885,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004768860000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0047808600000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00478965,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00290457,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00480119,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004775440000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00128322,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00063294,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0006368400000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0030546400000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0011165600000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.004690779,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.0013858400000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00017732000000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.00302739,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00025823,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0011078400000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0048495000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00035992,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.00480181,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.004800400000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0013250100000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00060264,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030313700000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00036419,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00034715,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00036321000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00035085000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00113582,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0029516300000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00291938,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004815300000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00110107,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0032178000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00036898000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00034919000000000004,
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
          "id": "1e62b5988e90deb790532c54823dbf41893dbc25",
          "message": "fix: disable 1.12 precompile (#1756)",
          "timestamp": "2025-10-15T19:49:48Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/1e62b5988e90deb790532c54823dbf41893dbc25"
        },
        "date": 1760587738717,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.001421083,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.0013329510000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.0029473890000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.003070105,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003044658,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.003051222,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.003012728,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.0031037680000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.001083241,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.003169878,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003035822,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.0014890340000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.0029112860000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0029697670000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.0030702430000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003008995,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003031656,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.001152629,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.0030616140000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.00304359,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.001349712,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030341780000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002072721,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0006791620000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0007978760000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007136109000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000688105,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029377450000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002084809,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00722961,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0029931840000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00034686,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0029678340000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029573000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.0010825000000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007198646,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007241,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007136883,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001085288,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006760090000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.00030547200000000005,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003219496,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.003226412,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0006662910000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012092162,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.002959727,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007161623000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00030172600000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0011190500000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.002934213,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.003361722,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007203745,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.003056557,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003151065,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007112212000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007113522000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.000310506,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0031178900000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000671295,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000670813,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007120447,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0006880850000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007151958000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.003101532,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.0072327550000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002076844,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030404250000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.0006595380000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002933054,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007110709000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006616500000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.0011722850000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002537035,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002112437,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007119678000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030038120000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002559212,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000681018,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0010784170000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0071210430000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0006804160000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.000660578,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002957173,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029168370000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0006657880000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.002075275,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.0010790910000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000310393,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002082668,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010308369000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.00070189,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.00031375,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029289370000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.002995081,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007235887000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0006854280000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004784001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0013158800000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0030501400000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0029348900000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00035210000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00039205,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0029204,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.00304666,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0048030600000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00035709,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00016831000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00477643,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00130542,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00305982,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00036212,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00035563,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00063144,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00304054,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0030406400000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00480343,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00110319,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00305308,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.0010241100000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.002639849,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00130775,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00016265000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.0003527,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.004790320000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00016373,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00132094,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030585100000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00060662,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0003395,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00015168,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00037047,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00481514,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0030657400000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00034906,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0047903600000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004806390000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00479965,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00293479,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00481318,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004805960000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00124544,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00062719,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0006105500000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00304958,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.00111705,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.00468684,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.0014071600000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00016373,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.0030537990000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00024391000000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0011078400000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0047894700000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.000346,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.004812251,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.004800969,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0013048900000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00061003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00305185,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00037356000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00034359,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00034763000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.0003436,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00113545,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00291545,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00291814,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0047994000000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00111443,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.00323587,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00038803000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00034115,
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
          "id": "66657521c818a0cc0c497645db15200c6f6b2140",
          "message": "chore: update ENZYMEXLA_COMMIT hash in WORKSPACE",
          "timestamp": "2025-10-17T00:22:09Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/66657521c818a0cc0c497645db15200c6f6b2140"
        },
        "date": 1760675188802,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0018862450000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.001772633,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.003953486,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.004046769,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0040304170000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.004002658,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.004018589,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.004004201000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.0018761700000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.004314895,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003986227,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.001890652,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.003870286,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004018345,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.0040875830000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004081121,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0039607010000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.001524502,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.0040050400000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.0041675570000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.001524511,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004050314,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0020674400000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0006575260000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0007911680000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007126923,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000676681,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029454850000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002070587,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007232247000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.002984252,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00035091500000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.002998712,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.002958379,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001078961,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.00722447,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007241483000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0071482460000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0010735620000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.00067332,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000320684,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003211854,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0032221430000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0006641,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012139047,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.00295678,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007151644,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.000316507,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0011382480000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.002948525,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.003379524,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0072394880000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.003058016,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.00318013,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007112506,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007132568000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00031618200000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003135039,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000665621,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000662,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0071221620000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0006806630000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007128156,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.003110443,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.0072575190000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002068674,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003081986,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.0006604250000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002925518,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007125265,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000664049,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001165227,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0025412240000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002105689,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007140110000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003017806,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002592759,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0006913920000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.00108466,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0071574360000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0006605000000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.000669099,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002930306,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029451580000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0006610850000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.002064691,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001075081,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00033864100000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002060379,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010239052,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000672946,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.00033537700000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002959157,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.002990379,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007217506,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000683434,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00479418,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0013175390000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00305645,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.00292865,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00035557000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00034826000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00292697,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0030215800000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.004799370000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00036157,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00015010000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00478136,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0013056,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00302051,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00035359000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00033948,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00061133,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0030351600000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0030596900000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00477186,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00108771,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00303842,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00101858,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00261435,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.0013086600000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00014745,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00033665,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00479657,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00015137,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00131375,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030372000000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00061746,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00033535000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00014865,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00034666000000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00478735,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0030375000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00034563000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004781280000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004773679,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00478323,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00292318,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004782430000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0047891,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00123558,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0006284400000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0006135400000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0030299790000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0010917700000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.00468142,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.0013588600000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00014886,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.0030535700000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00025932,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0010827,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00480563,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00033749,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.00478955,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0048111000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00131101,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00061328,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030461390000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00034361,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00034305,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00035635,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00033609000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0011233500000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0029159000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00290857,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00479877,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.0010892010000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0032014,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00033981,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00036113000000000003,
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
          "id": "d92c0ef8d71136459c4e3d9cbc3d87c441d652fa",
          "message": "Regenerate MLIR Bindings (#1769)\n\nCo-authored-by: enzyme-ci-bot[bot] <78882869+enzyme-ci-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-18T01:17:17Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/d92c0ef8d71136459c4e3d9cbc3d87c441d652fa"
        },
        "date": 1760758307473,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.001472245,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.001207844,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.003104772,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.003252777,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0033930770000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.0031572400000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.0033411020000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.00340395,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.001331692,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.003546445,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0034265830000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.001325201,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.0032254040000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0033130760000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.003436798,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0034302720000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003336475,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.0012447080000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.0033315830000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.0035120610000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.001269588,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0033123880000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002065495,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0006765100000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0008032930000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007170305,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000694977,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.002984103,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0020720450000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007274697,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0029858190000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00035403100000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.002990485,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029475960000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.0010775440000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007254801000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007235507,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007168306,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001082917,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006634420000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000318777,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0032321940000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.003238653,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000683027,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012242069000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.002961703,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007197773,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00030829100000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0011402580000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0029957960000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0034161860000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007264354000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0030743000000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0032027080000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007156553,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007167095,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.00031473000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0031316200000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0006615600000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0006698180000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007158336,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00069172,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.00716459,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0031301790000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.0073026390000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002109447,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003070095,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000665364,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0029510030000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007157318,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006724560000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.00118219,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0025371210000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002120889,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007173552000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.003001796,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002570399,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000697523,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001085992,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007156396000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0006678410000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0006723030000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0029710860000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.002956979,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0006722010000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.0020893640000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001085388,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000317923,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0020793070000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010351951,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000694857,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.00032281700000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002950836,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003009494,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007240662,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000692601,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0047921000000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00131403,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00306084,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0029460600000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00036056,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000363759,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0029335900000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0030631200000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0048091,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00035979,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00017926000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0047904,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0013305600000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030752400000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00037076000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00037641000000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00062905,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00305185,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0030447300000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.004788870000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00112392,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00304914,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00100032,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00261368,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.0013161100000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00017847000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00036923,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.004788580000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.0001708,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.001318881,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00305214,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0006351400000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00036577,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00017391000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00037415,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0047942900000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00305321,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00035742,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00478665,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00481248,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00478822,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0029430100000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0047926800000000006,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0047976,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.0012455410000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00063121,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.000626,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0030583900000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0011422300000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.00466611,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.0014105300000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00015627000000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.003073099,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00028687,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00111918,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00479745,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00036918000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.00477423,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00480427,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00130903,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0006212500000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030741400000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00036714,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0003621,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00035889,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00035952,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0011612500000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0029383900000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00294092,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0047942,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.001121631,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.00323862,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0003668,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00036269000000000004,
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
          "id": "6aa98ad5e3a93bcce3d82180007b51ea330d67b1",
          "message": "fix: TracedRNG path for correct create_result (#1803)",
          "timestamp": "2025-10-30T00:17:04Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/6aa98ad5e3a93bcce3d82180007b51ea330d67b1"
        },
        "date": 1761795910239,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0025661300000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.002363525,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.005805968000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.0060180730000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.005777667,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.0058535670000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.006026662,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.005929626,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.002280026,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.006073655000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.005882186,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.002438252,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.005600801000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.005914184,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.005913824000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.005908058000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0056625140000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.002313213,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.005955371,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.005901105,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.0024102380000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0060392350000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0020559230000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000669692,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0007995190000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007196404000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000705605,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029370340000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0020536860000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007298158000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0029878810000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.000357001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.002992236,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0029416470000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001085921,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007313845,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007300839000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.007159067000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001084041,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.000661446,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000324083,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003167188,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0031743,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0007744570000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012526184000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0029275190000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007172351,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.000326448,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0011326390000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.002958945,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0034767260000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007241286,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.003069828,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0031388690000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0071183390000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007141949000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.000330755,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0031230190000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0006870520000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0006858330000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007137023,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00071646,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007161197,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.003079352,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.0072989710000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0020463810000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003061746,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000683309,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0029476190000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0071447540000000006,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0006861290000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.0011729630000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0024770160000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002109225,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007158305,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029808110000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002515948,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.000701088,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001091035,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007165141000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0006954960000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0006647640000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0029302990000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0029363080000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000653542,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.0019856960000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.0010843580000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.000337014,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0020462,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010634199,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.00069393,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.00033843100000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002936475,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.002964705,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007260687,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.000703691,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004843858,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0013721800000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.003074189,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0029780790000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00037401000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00037600000000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0029668090000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0030868090000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.004866268,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00038322000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00020027,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.004846398,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.001371129,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003060019,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00037974000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00037553,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0006402900000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0030586190000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0030578790000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.004823288,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.001132449,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.003081219,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00107762,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.002673589,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00137512,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00020901,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00040696000000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.004827518,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.000210019,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00137494,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003085249,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.000649719,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.000379539,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00020848,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00041736000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.004853218,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0030723580000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00038017000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004827598000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004849058000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.004814238,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.002945809,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004834819000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.004831819,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.001297669,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00066562,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.00064198,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0030492490000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.00112633,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.004774878000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00141803,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00019546,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.003073749,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00032224900000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00112896,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.004848488000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00038218000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.004829458,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.004838328,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0013681790000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00064618,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003101808,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00037908,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00038242,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.000381479,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00037235,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.001163189,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.002975639,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002952868,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004852048,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00113282,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0032791290000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00043036,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00037989,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "William Moses",
            "username": "wsmoses",
            "email": "wmoses@google.com"
          },
          "committer": {
            "name": "William Moses",
            "username": "wsmoses",
            "email": "gh@wsmoses.com"
          },
          "id": "e5955c83dbf6dc1ec8be193a894cd97ea92e794a",
          "message": "Apply suggestions from code review\n\nCo-authored-by: github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-30T22:53:03Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/e5955c83dbf6dc1ec8be193a894cd97ea92e794a"
        },
        "date": 1761902046014,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0018775150000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.001623918,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.003945208,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.004161944,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004399468,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.004339074,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.004322162,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.004210979,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.0016762390000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.004353646,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0040494680000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.001988566,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.004205889,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.0040771010000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.004189913,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004204076,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004224969,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.001584941,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.0041887800000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.004288104,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.0015446440000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.004465847,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.0020547390000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000687629,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.000765975,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007259917,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.000705317,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.0029913,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0020580390000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007334613,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0030047330000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.000403318,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.002993122,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.002971411,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.001092717,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007287751,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007278636000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.00720813,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.001091657,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.000661082,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000324254,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0031805920000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.003201661,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.000687456,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012547355000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0029308760000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007283149,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.000344796,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.001138165,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0029407960000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0033663670000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007271316000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0030784730000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003149853,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0071923740000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007200988,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.000349574,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003126764,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0006886290000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0006908610000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007141824000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.000716162,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.0072163850000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.0031196070000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007343425000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.0020634860000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003119988,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.000674285,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0029504540000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.007224454000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000684955,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001181807,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002493925,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002120116,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007195778000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0029834790000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002489189,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0007145550000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0010836160000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007227320000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.0006731360000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0006695630000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.002965605,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.002940762,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000666377,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.002060192,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001094093,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00033825200000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.002073234,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010799122000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.0006990150000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000375385,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.002991243,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0029962310000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007304526,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0007085,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00480986,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00135293,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.0030602800000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.00293952,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00037996000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00035752,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00295129,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0031003800000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.004806970000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00037876,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00019972000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00482646,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00133678,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00304951,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00038999,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00038814000000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.0006226700000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0030568600000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00304467,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00483374,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00113422,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0030798600000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00104097,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00261631,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.00135193,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00019981,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00037931,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00482136,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00020369000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00137004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00306383,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0006161700000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00040167,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00020466,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00039548000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00482819,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0030638,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00038879,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.004806381,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00483141,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00479855,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00295011,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00482527,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00481483,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00127916,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0006308300000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.00063142,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00307894,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.0011304000000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.004728361,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.0014166600000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00020512000000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.00308294,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00030803000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00113331,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.004811070000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00038241,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.00481465,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.004814570000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0013578400000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0006149700000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0030705800000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00036598,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00039232000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00038759000000000005,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00036935,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0011625700000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.0029431400000000003,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029503900000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00482675,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.00112811,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.00326762,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00036492,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00038432,
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
          "id": "ccd6fb985b86be4f2d739a238ba4029e1ae0da35",
          "message": "fix: ensure randn respects exclusivity of maxval (#1815)\n\n* fix: ensure randn respects exclusivity of maxval\n\n* feat: allow traced inputs",
          "timestamp": "2025-10-31T22:34:53Z",
          "url": "https://github.com/EnzymeAD/Reactant.jl/commit/ccd6fb985b86be4f2d739a238ba4029e1ae0da35"
        },
        "date": 1761968595098,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/Default",
            "value": 0.0014683270000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGatherPad",
            "value": 0.0012234280000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAfterEnzyme",
            "value": 0.003234625,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAfterEnzyme",
            "value": 0.00320598,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.003401026,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAll",
            "value": 0.003219248,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadBeforeEnzyme",
            "value": 0.003424836,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisablePadAll",
            "value": 0.0033647300000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableScatterGather",
            "value": 0.0013387380000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultAll",
            "value": 0.0034370810000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0032356700000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/XLA",
            "value": 0.0014444590000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/XLA",
            "value": 0.003237465,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAfterEnzyme",
            "value": 0.003243681,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DefaultBeforeEnzyme",
            "value": 0.003376083,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003279309,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.003195644,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisablePad",
            "value": 0.0013365150000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableTransposeReshapeAll",
            "value": 0.0032559010000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherAll",
            "value": 0.0033241250000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CPU/DisableTransposeReshape",
            "value": 0.001335601,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.003380974,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002052959,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAll",
            "value": 0.000692777,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/XLA",
            "value": 0.0008147580000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.007170391000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.00068592,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAll",
            "value": 0.002939054,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.002049333,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.007301037000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.0030014060000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/XLA",
            "value": 0.00037180400000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.002976513,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.002931364,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisablePad",
            "value": 0.0010865780000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.007394943,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.007249659,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0071684850000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.0010920980000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAll",
            "value": 0.0006789890000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableTransposeReshape",
            "value": 0.000325996,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGatherPad",
            "value": 0.003134988,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0032196380000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.0006776270000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/XLA",
            "value": 0.012523844000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.0029594150000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.007165254,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGather",
            "value": 0.00033220300000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.00115346,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAll",
            "value": 0.0029598370000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/XLA",
            "value": 0.0033923610000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.007274414000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.0030586100000000002,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.003174108,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAll",
            "value": 0.007153139,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.007225986,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisablePad",
            "value": 0.000316136,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.003113441,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherAfterEnzyme",
            "value": 0.000678925,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.000696333,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.007147113,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0007016430000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.007165523,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/XLA",
            "value": 0.003131603,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/XLA",
            "value": 0.007280872000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableTransposeReshape",
            "value": 0.002067124,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.003091892,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DefaultAll",
            "value": 0.0006603360000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.002922688,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.00717846,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.000674641,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/XLA",
            "value": 0.001180395,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/DisablePad",
            "value": 0.002495115,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/XLA",
            "value": 0.002112025,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.007148850000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.002992731,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/CUDA/Default",
            "value": 0.002530905,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherBeforeEnzyme",
            "value": 0.0006971210000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.001097572,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DisableTransposeReshapeAfterEnzyme",
            "value": 0.007189636,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadAfterEnzyme",
            "value": 0.000700917,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadAll",
            "value": 0.0006955300000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherAll",
            "value": 0.0029467710000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DefaultAfterEnzyme",
            "value": 0.002951066,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableTransposeReshapeAll",
            "value": 0.000659472,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/Default",
            "value": 0.001971584,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/CUDA/Default",
            "value": 0.001085538,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/DisableScatterGatherPad",
            "value": 0.00033739400000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/CUDA/DisableScatterGather",
            "value": 0.0020872810000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/CUDA/DefaultAll",
            "value": 0.010316071000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisablePadBeforeEnzyme",
            "value": 0.000691545,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/CUDA/Default",
            "value": 0.000329778,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadAfterEnzyme",
            "value": 0.0029396860000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.002992138,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/CUDA/DefaultBeforeEnzyme",
            "value": 0.007245869,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/CUDA/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0006940340000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.004815730000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisablePad",
            "value": 0.0013648100000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00308974,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/XLA",
            "value": 0.0029509000000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.0003397,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.00033106000000000005,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00296429,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAll",
            "value": 0.0030638700000000002,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00479158,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.00033610000000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/Default",
            "value": 0.00019619,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00481217,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00135723,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.0030713800000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00033476000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00033337000000000004,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00060547,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0030589000000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.0030522500000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00479819,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0011136,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadBeforeEnzyme",
            "value": 0.00307705,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/XLA",
            "value": 0.00100976,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.0026137300000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/Default",
            "value": 0.0013661,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGatherPad",
            "value": 0.00018494,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAll",
            "value": 0.00033824,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultAll",
            "value": 0.00480324,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableScatterGather",
            "value": 0.00019472,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0013651910000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.0030895700000000003,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisablePad",
            "value": 0.00060756,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0003326,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisablePad",
            "value": 0.00019329000000000001,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/XLA",
            "value": 0.00035502,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.0047900500000000006,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.0031120400000000004,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAll",
            "value": 0.00035045000000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00478806,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableTransposeReshapeAfterEnzyme",
            "value": 0.004794720000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAll",
            "value": 0.00480221,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.0029578200000000003,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.0048024800000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherAfterEnzyme",
            "value": 0.00483086,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/XLA",
            "value": 0.00127241,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.0006199700000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/Default",
            "value": 0.0006034100000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.00308223,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/Default",
            "value": 0.00112088,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/XLA",
            "value": 0.00474076,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/XLA",
            "value": 0.00138494,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/DisableTransposeReshape",
            "value": 0.00019406000000000002,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisablePadAll",
            "value": 0.00309411,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/forward/TPU/XLA",
            "value": 0.00026781,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableScatterGatherPad",
            "value": 0.00111749,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00480206,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00035358000000000004,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisablePadAll",
            "value": 0.004824060000000001,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DefaultBeforeEnzyme",
            "value": 0.0047976600000000005,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.0013488600000000001,
            "unit": "s"
          },
          {
            "name": "ViT tiny [256, 256, 3, 4]/forward/TPU/DisableScatterGather",
            "value": 0.00059455,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherBeforeEnzyme",
            "value": 0.00308864,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeBeforeEnzyme",
            "value": 0.00032987,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00033406,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAll",
            "value": 0.00033492,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableScatterGatherAll",
            "value": 0.00033872000000000004,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisableTransposeReshape",
            "value": 0.00114171,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DefaultAfterEnzyme",
            "value": 0.00295857,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/DisableScatterGatherPadAfterEnzyme",
            "value": 0.00295837,
            "unit": "s"
          },
          {
            "name": "VGG11 bn=true [224, 224, 3, 4]/backward/TPU/DisableScatterGatherPadBeforeEnzyme",
            "value": 0.004812530000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/forward/TPU/DisablePad",
            "value": 0.0011066700000000001,
            "unit": "s"
          },
          {
            "name": "FNO [64, 64, 1, 4]/backward/TPU/XLA",
            "value": 0.0032519100000000002,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisableTransposeReshapeAll",
            "value": 0.00034041000000000003,
            "unit": "s"
          },
          {
            "name": "DeepONet ([64, 1024], [1, 128])/backward/TPU/DisablePadAfterEnzyme",
            "value": 0.00032431,
            "unit": "s"
          }
        ]
      }
    ]
  }
}