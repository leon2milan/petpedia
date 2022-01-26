pipeline {
    agent any

    stages {
        stage('Preprocess') {
            steps {
                echo "Switching user..."
                sh 'cp /home/www/switch_user.exp ./' 
                sh 'expect switch_user.exp' 
                echo "Switching directory..."
                sh 'cd /workspaces/ai-petpedia'
                echo "Activating env..."
                sh '''#!/usr/bin/env bash
                source /home/jiangbingyu/miniconda3/bin/activate petpedia
                which python'''
                sh 'which python'
            }
        }
        stage('git checkout'){
            steps {
                git branch: 'master',
                    credentialsId: '154a2ac7-3cc8-4c7b-a587-1ef55ac5e9d6',
                    url: 'https://git.rp-field.com/ai/ai-petpedia'

            }   
        }
        stage('Build') {
            steps {
                echo 'Building...'
                sh '''#!/usr/bin/env bash
                source /home/jiangbingyu/miniconda3/bin/activate petpedia
                make build'''
            }
        }
        stage('Test') {
            steps {
                echo 'Testing...'
                sh '''#!/usr/bin/env bash
                source /home/jiangbingyu/miniconda3/bin/activate petpedia
                make test'''
            }
        }
        // stage('Deploy') {
        //     steps {
        //         echo 'Deploying...',
        //         sh '''#!/usr/bin/env bash
        //         source /home/jiangbingyu/miniconda3/bin/activate petpedia
        //         make run'''
        //     }
        // }
    }
}