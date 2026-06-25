from src.pipeline.rag_pipeline import run_rag_pipeline


def print_pipeline_result(result):
    print("\nQuestion:")
    print(result["question"])

    if not result["success"]:
        print("\nError:")
        print(result["error"])
        return

    print("\nCache hit:")
    print(result.get("cache_hit"))

    print("\nLatency:")
    print(f"{result.get('latency_seconds')} seconds")

    print("\nTimings:")
    print(result.get("timings"))

    print("\nReference title:")
    print(result.get("reference_title"))

    print("\nDetected filters:")
    print(result.get("filters"))

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(
            f"- {source['title']} "
            f"({source['type']}, {source['year']}, {source['rating']})"
        )


if __name__ == "__main__":
    # question = ["thriller shows like Dark","Indian comedy movies","kids movies about animals","Korean horror shows","documentaries about nature"]
    # for q in question:
    #     result = run_rag_pipeline(q)
    #     print_pipeline_result(result)
    query = "thriller shows like Dark"

    print("\n========== First Run ==========")
    result = run_rag_pipeline(query)
    print("\nCache hit:")
    print(result.get("cache_hit"))

    print("\nCache type:")
    print(result.get("cache_type"))

    print("\nLatency:")
    print(result.get("latency_seconds"))
    print_pipeline_result(result)

    print("\n========== Second Run ==========")
    result_2 = run_rag_pipeline(query)
    print_pipeline_result(result_2)