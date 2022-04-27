functionPlot({
    target: "#function1",
    width,
    height,
    yAxis: { domain: [-1, 9] },
    grid: true,
    data: [
      {
        fn: "x^2",
        derivative: {
          fn: "2 * x",
          updateOnMouseMove: true
        }
      }
    ]
  });