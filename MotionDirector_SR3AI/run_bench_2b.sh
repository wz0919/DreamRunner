trap 'kill 0' SIGINT

bash t2v-combench-2b.sh "1_consistent_attribute_binding"
bash t2v-combench-2b.sh "2_dynamic_attribute_binding"
bash t2v-combench-2b.sh "3_spatial_relationships"
bash t2v-combench-2b.sh "4_motion_binding"
bash t2v-combench-2b.sh "5_action_binding"
bash t2v-combench-2b.sh "6_object_interactions"