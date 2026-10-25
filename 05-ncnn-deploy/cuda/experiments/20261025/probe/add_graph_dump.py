#!/usr/bin/env python3
"""H-048: optional logits dump from the graph path (env NCNN_CUDA_GRAPH_DUMP)."""
import io
import sys

CPP = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_graph_eval.cpp"

OLD = """                        cudaEvent_t e0, e1;
                        cudaEventCreate(&e0);
                        cudaEventCreate(&e1);
                        cudaEventRecord(e0, gs);
"""

NEW = """                        const char* dump_path = std::getenv("NCNN_CUDA_GRAPH_DUMP");
                        if (dump_path)
                        {
                            FILE* fp = fopen(dump_path, "w");
                            if (fp)
                            {
                                for (int j = 0; j < n; j++)
                                {
                                    fprintf(fp, "%d %d", j, (int)batch_data.at(j).label);
                                    for (int k = 0; k < 10; k++)
                                        fprintf(fp, " %a", outs[(size_t)j * 10 + k]);
                                    fprintf(fp, "\\n");
                                }
                                fclose(fp);
                            }
                        }

                        cudaEvent_t e0, e1;
                        cudaEventCreate(&e0);
                        cudaEventCreate(&e1);
                        cudaEventRecord(e0, gs);
"""

text = io.open(CPP, encoding="utf-8").read()
if OLD not in text:
    sys.exit("dump anchor missing")
text = text.replace(OLD, NEW, 1)
io.open(CPP, "w", encoding="utf-8", newline="\n").write(text)
print("dump hook inserted")
