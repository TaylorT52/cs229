## Setup 

### How this works
* define a configuration file that defines vehicles, IDM controllers, and defines vehicles
    * IDM controllers = how the car accelerates and decelerates 
    * continuousRoutes keep the car on the highway
    * Sumo params and environment params
    * The test environment (note - this can change when we decide to create custom networks)

* then, define a file that runs the experiment
    * all this does is set python paths, modules
    * imports the configurations
    * creates and experiment and runs it