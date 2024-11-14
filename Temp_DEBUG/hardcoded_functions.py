+import time^M
+import threading^M
+from tts import generate_audio^M
+from tts_run import main, split_text, generate_audio, combined_audio^M
+from stt import stt_model, process_audio_in_chunks^M
+from stt_tour_workflow import record_audio^M
+from Interface.rag_pll import user_query^M
+from Wake_word import wake_word_callback_no, wake_word_callback_yes, listen_for_wake_word^M
+^M
+^M
+^M
+^M
+# Simulating get_info function without caching^M
+def get_info(location_tag):^M
+    start_time = time.time()^M
+    # Simulate a fetch operation (external API, database, etc.)^M
+    time.sleep(1)  # Simulated delay for fetching data^M
+    end_time = time.time()^M
+    print(f"get_info execution time: {end_time - start_time:.2f} seconds for location: {location_tag}")^M
+    return^M
+^M
+# Simulate changing states with a sequence^M
+# state_sequence = [1, 1, 0, 1, 0, 0]^M
+# state_index = 0^M
+^M
+def get_state():^M
+    # global state_index^M
+    # # start_time = time.time()^M
+    # # start_time = time.time()^M
+^M
+    # # Simulate state retrieval^M
+    # state = state_sequence[state_index % len(state_sequence)]^M
+    # state_index += 1^M
+^M
+    # Reduced the sleep time to a more efficient check^M
+    time.sleep(4)  # Reduced delay for state retrieval^M
+    # end_time = time.time()^M
+    # print(f"get_state execution time: {end_time - start_time:.2f} seconds")^M
+    return^M
+    # return state^M
+^M
+def monitor_state():^M
+    # start_time = time.time()^M
+    previous_state = None^M
+    backoff_time = 0.1  # Initial polling interval (dynamically adjusted)^M
+^M
+    while True:  # Infinite loop for continuous monitoring^M
+        current_state = get_state()^M
+        # print(f"Previous State: {previous_state}, Current State: {current_state}")^M
+^M
+        if previous_state == 1 and current_state == 0:^M
+            location_tag = "exhibit_x"  #Represents a hypothetical exhibit for which the information is to be loaded^M
+            # Run get_info in a separate thread to avoid blocking^M
+            info_thread = threading.Thread(target=get_info, args=(location_tag,))^M
+            info_thread.start()^M
+        # This fix ^M
+        previous_state = current_state
+        # Dynamically increase sleep time if no state change (to avoid excessive polling)^M
+        if previous_state == current_state:^M
+            backoff_time = min(backoff_time + 0.1, 1)  # Gradually increase polling interval up to 1 second^M
+        else:^M
+            backoff_time = 0.1  # Reset backoff if state changes^M
+^M
+        time.sleep(backoff_time)^M
+^M
+    # end_time = time.time()^M
+    # print(f"monitor_state execution time: {end_time - start_time:.2f} seconds")^M
+^M
+# def ask_further_questions(question_count=0, max_questions=5):^M
+#     """Guide robot asks whether the user has further questions using TTS and processes the response using STT.^M
+#        The robot will stop asking after 5 questions."""^M
+    ^M
+#     # start_time = time.time()^M
+^M
+#     if question_count >= max_questions:^M
+#         last = "I hope I have been able to answer your questions well. Let us now continue with our tour."^M
+#         main(last)^M
+#         # end_time = time.time()^M
+#         # print(f"ask_further_questions execution time: {end_time - start_time:.2f} seconds")^M
+#         return  ^M
+^M
+#     question = 'Do you have any questions? Please answer \'Yes I do\' or \'No I don\'t\'.' if question_count == 0 else 'Do you have any further questions? Please answer \'Yes I do\' or \'No I don\'t\'.'^M
+    ^M
+#     # Simulate speaking the question and waiting for response in parallel^M
+#     main(question)^M
+^M
+#     # recording_duration_ms = 10000^M
+#     # yes_no_audio_file = record_audio(recording_duration_ms)^M
+#     # yes_no_text = process_audio_in_chunks(yes_no_audio_file, chunk_length_ms=10000)^M
+^M
+#     yes_no = listen_for_wake_word()^M
+^M
+#     if yes_no == 'yes':^M
+#         main("Please go ahead")^M
+#         input_question = record_audio()^M
+        ^M
+#         if input_question == "Sorry, I couldn't understand that.":^M
+#             # print(follow_up_question)^M
+#             answer_text = RAG(input_question)  # Simulate RAG answer generation and speaking the answer^M
+#             main(answer_text)^M
+#             ask_further_questions(question_count + 1, max_questions)  ^M
+#         elif:^M
+#             main("Sorry, I did not understand your question.")^M
+#             ask_further_questions(question_count, max_questions)  ^M
+            ^M
+#     elif yes_no_text == 'no' or yes_no_text == 'No' or yes_no_text == 'no.' or yes_no_text == 'No.':^M
+#         time.sleep(4)^M
+#         main("Okay then, let us go ahead with our tour!")^M
+        ^M
+#     else:^M
+#         main("I did not catch that. Could you please respond with yes or no?")^M
+#         time.sleep(5)^M
+#         ask_further_questions(question_count, max_questions)   ^M
+^M
+^M
+^M
+def ask_further_questions(question_count=0, max_questions=5):^M
+    """Guide robot asks whether the user has further questions using TTS and processes the response using STT.^M
+       The robot will stop asking after 5 questions."""^M
+^M
+    if question_count >= max_questions:^M
+        last = "I hope I have been able to answer your questions well. Let us now continue with our tour."^M
+        main(last)^M
+        return  ^M
+^M
+    question = 'Do you have any questions? Please answer \'Yes I do\' or \'No I don\'t\'.' if question_count == 0 else 'Do you have any further questions? Please answer \'Yes I do\' or \'No I don\'t\'.'^M
+    ^M
+    # Speak the question and wait for the response^M
+    main(question)^M
+    yes_no = listen_for_wake_word()^M
+^M
+    if yes_no == 'yes':^M
+        main("Please go ahead")^M
+        retry_question(question_count, max_questions)^M
+        ^M
+    elif yes_no == 'no':^M
+        time.sleep(3)^M
+        main("Okay then, let us go ahead with our tour!")^M
+    else:^M
+        main("I did not catch that. Could you please respond with \'Yes I do\' or \'No I don\'t\'?")^M
+        time.sleep(1)^M
+        ask_further_questions(question_count, max_questions)  ^M
+^M
+^M
+def retry_question(question_count, max_questions, retry_limit=3):^M
+    """Retries to capture the user's question if STT fails to understand it, up to retry_limit attempts."""^M
+    ^M
+    retries = 0^M
+    while retries < retry_limit:^M
+        input_question = record_audio()^M
+        ^M
+        if "Sorry, I couldn't understand that." in input_question:^M
+            main("Could you please repeat your question?")^M
+            retries += 1^M
+        else:^M
+            # Successfully captured question, generate and provide response^M
+            answer_text = user_query(input_question)  # Simulate RAG answer generation and speak the answer^M
+            main(answer_text)^M
+            ask_further_questions(question_count + 1, max_questions)^M
+            return  # Exit the function once a valid question is handled^M
+    ^M
+    # If retry limit is reached without a clear question^M
+    main("Sorry, I did not understand your question.")^M
+    ask_further_questions(question_count, max_questions)^M
+ ^M
+^M
+monitor_state()