FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list
RUN apt-get update && apt-get install -y git cmake wget build-essential
RUN git clone https://gitee.com/bingyu_jiang/ai_guidance.git /workspaces/aiguidance
RUN make install
