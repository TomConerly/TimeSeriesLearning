def d3script(csvPath):
    start = '''
<!DOCTYPE html>
<meta charset="utf-8">
<style>
 
svg {
  font: 10px sans-serif;
}
 
 
.axis path,
.axis line {
  fill: none;
  stroke: #000;
  shape-rendering: crispEdges;
}

.y.axis path {
  fill: none;
  stroke: #000;
  shape-rendering: crispEdges;
}

.brush .extent {
  stroke: #fff;
  fill-opacity: .125;
  shape-rendering: crispEdges;
}

.line {
  fill: none;
}

</style>
<body>
<script src="http://d3js.org/d3.v3.min.js"></script>
<script>
 
var margin = {top: 10, right: 10, bottom: 100, left: 40},
    margin2 = {top: 430, right: 10, bottom: 20, left: 40},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom,
    height2 = 500 - margin2.top - margin2.bottom;
 
var color = d3.scale.category10();
 
var x = d3.scale.linear().range([0, width]),
    x2 = d3.scale.linear().range([0, width]),
    y = d3.scale.linear().range([height, 0]),
    y2 = d3.scale.linear().range([height2, 0]);
 
var xAxis = d3.svg.axis().scale(x).orient("bottom"),
    xAxis2 = d3.svg.axis().scale(x2).orient("bottom"),
    yAxis = d3.svg.axis().scale(y).orient("left");
 
var brush = d3.svg.brush()
    .x(x2)
    .on("brush", brush);
 
var line = d3.svg.line()
    .defined(function(d) { return true; })
    .interpolate("cubic")
    .x(function(d) { return x(d.step); })
    .y(function(d) { return y(d.value); });
 
var line2 = d3.svg.line()
    .defined(function(d) { return true; })
    .interpolate("cubic")
    .x(function(d) {return x2(d.step); })
    .y(function(d) {return y2(d.value); });
 
var svg = d3.select("body").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom);
 
svg.append("defs").append("clipPath")
    .attr("id", "clip")
    .append("rect")
    .attr("width", width)
    .attr("height", height);
 
var focus = svg.append("g")
  .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var context = svg.append("g")
  .attr("transform", "translate(" + margin2.left + "," + margin2.top + ")");

var sources = [];
'''
    mid = 'd3.csv("' + csvPath+ '", function(error, data) {'
    end = '''

    color.domain(d3.keys(data[0]).filter(function(key) { return key !== "step"; }));

    sources = color.domain().map(function(name) {
      return {
        name: name,
        values: data.map(function(d) {
          return {step: +d.step, value: +d[name]};
        })
      };
    });

    x.domain(d3.extent(data, function(d) { return +d.step; }));
    y.domain([d3.min(sources, function(c) { return d3.min(c.values, function(v) { return v.value; }); }),
              d3.max(sources, function(c) { return d3.max(c.values, function(v) { return v.value; }); }) ]);
    x2.domain(x.domain());
    y2.domain(y.domain());

    var focuslineGroups = focus.selectAll("g")
        .data(sources)
        .enter().append("g");

    var focuslines = focuslineGroups.append("path")
        .attr("class","line")
        .attr("d", function(d) { return line(d.values); })
        .style("stroke", function(d) {return color(d.name);})
        .attr("clip-path", "url(#clip)");

    focus.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

    focus.append("g")
        .attr("class", "y axis")
        .call(yAxis);

    var contextlineGroups = context.selectAll("g")
        .data(sources)
        .enter().append("g");

    var contextLines = contextlineGroups.append("path")
        .attr("class", "line")
        .attr("d", function(d) { return line2(d.values); })
        .style("stroke", function(d) {return color(d.name);})
        .attr("clip-path", "url(#clip)");

    context.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height2 + ")")
        .call(xAxis2);

    context.append("g")
        .attr("class", "x brush")
        .call(brush)
        .selectAll("rect")
        .attr("y", -6)
        .attr("height", height2 + 7);

    var mouseG = svg.append("g")
      .attr("class", "mouse-over-effects");

    mouseG.append("path") // this is the black vertical line to follow mouse
      .attr("class", "mouse-line")
      .style("stroke", "black")
      .style("stroke-width", "1px")
      .style("opacity", "0");

    var lines = document.getElementsByClassName('line');

    var mousePerLine = mouseG.selectAll('.mouse-per-line')
      .data(sources)
      .enter()
      .append("g")
      .attr("class", "mouse-per-line");

    mousePerLine.append("circle")
      .attr("r", 7)
      .style("stroke", function(d) {
        return color(d.name);
      })
      .style("fill", "none")
      .style("stroke-width", "1px")
      .style("opacity", "0");

    mousePerLine.append("text")
      .attr("transform", "translate(10,3)");

    mouseG.append('svg:rect') // append a rect to catch mouse movements on canvas
      .attr('width', width) // can't catch mouse events on a g element
      .attr('height', height)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .on('mouseout', function() { // on mouse out hide line, circles and text
        d3.select(".mouse-line")
          .style("opacity", "0");
        d3.selectAll(".mouse-per-line circle")
          .style("opacity", "0");
        d3.selectAll(".mouse-per-line text")
          .style("opacity", "0");
      })
      .on('mouseover', function() { // on mouse in show line, circles and text
        d3.select(".mouse-line")
          .style("opacity", "1");
        d3.selectAll(".mouse-per-line circle")
          .style("opacity", "1");
        d3.selectAll(".mouse-per-line text")
          .style("opacity", "1");
      })
      .on('mousemove', function() { // mouse moving over canvas
        var mouse = d3.mouse(this);
        d3.select(".mouse-line")
          .attr("d", function() {
            var d = "M" + mouse[0] + "," + height;
            d += " " + mouse[0] + "," + 0;
            return d;
          });

        d3.selectAll(".mouse-per-line")
          .attr("transform", function(d, i) {
            console.log(width/mouse[0])
            var xDate = x.invert(mouse[0]),
                bisect = d3.bisector(function(d) { return d.date; }).right;
                idx = bisect(d.values, xDate);

            var beginning = 0,
                end = lines[i].getTotalLength(),
                target = null;

            while (true){
              target = Math.floor((beginning + end) / 2);
              pos = lines[i].getPointAtLength(target);
              if ((target === end || target === beginning) && pos.x !== mouse[0]) {
                  break;
              }
              if (pos.x > mouse[0])      end = target;
              else if (pos.x < mouse[0]) beginning = target;
              else break; //position found
            }

            d3.select(this).select('text')
              .text(y.invert(pos.y).toFixed(2));

            return "translate(" + mouse[0] + "," + pos.y +")";
          });
      });

});

function brush() {
  var domain = brush.empty() ? x2.domain() : brush.extent();
  x.domain(domain);
  y.domain([d3.min(sources, function(c) { return d3.min(c.values, function(v) { return domain[0] <= v.step && v.step <= domain[1] ? v.value : 1; }); }),
            d3.max(sources, function(c) { return d3.max(c.values, function(v) { return domain[0] <= v.step && v.step <= domain[1] ? v.value : 0; }); }) ]);
  focus.selectAll("path.line").attr("d",  function(d) {return line(d.values)});
  focus.select(".x.axis").call(xAxis);
  focus.select(".y.axis").call(yAxis);
}
 
</script>
'''
    return start + mid + end
