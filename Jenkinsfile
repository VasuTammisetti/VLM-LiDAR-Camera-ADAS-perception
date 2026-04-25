stage('Unit Tests') {
            steps {
                sh '''
                    set +e
                    docker run --name vlm-adas-test-run \
                        ${DOCKER_IMAGE_TEST} \
                        pytest tests/ -v --tb=short --junitxml=/app/results.xml
                    EXIT_CODE=$?

                    mkdir -p test-results
                    docker cp vlm-adas-test-run:/app/results.xml test-results/results.xml || true
                    docker rm vlm-adas-test-run || true

                    exit $EXIT_CODE
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results/results.xml'
                }
            }
        }