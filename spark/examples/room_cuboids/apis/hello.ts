const helloPlugin = {
    name: "hello",
    setup() {
        console.log("Hello WebXR!")
    }, 
    configureServer(server) {
        console.log("Configuring server with hello plugin")
        server.middleware.use((req, res, next) => {
            console.log("Hello from server middleware!")

            

            next()
        })
    }
}

export default helloPlugin;