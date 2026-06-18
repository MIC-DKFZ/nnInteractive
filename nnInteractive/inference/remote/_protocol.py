"""Shared constants for the nnInteractive client/server HTTP protocol."""

# HTTP header used to carry JSON-encoded metadata alongside a binary array body.
META_HEADER = "X-Meta"
# HTTP header used to carry a per-client lease token identifying which session
# on the (multi-session) server the request applies to.
LEASE_HEADER = "X-Lease-Token"

# Endpoint paths.
PATH_HEALTHZ = "/healthz"
PATH_CAPABILITIES = "/capabilities"
PATH_CLAIM = "/claim"
PATH_RELEASE = "/release"
PATH_HEARTBEAT = "/heartbeat"
PATH_LEASE_STATUS = "/lease_status"
PATH_SET_IMAGE = "/set_image"
PATH_SET_TARGET_BUFFER = "/set_target_buffer"
PATH_RESET_INTERACTIONS = "/reset_interactions"
PATH_UNDO = "/undo"
PATH_SET_DO_AUTOZOOM = "/set_do_autozoom"
PATH_ADD_BBOX = "/add_bbox_interaction"
PATH_ADD_POINT = "/add_point_interaction"
PATH_ADD_SCRIBBLE = "/add_scribble_interaction"
PATH_ADD_LASSO = "/add_lasso_interaction"
PATH_ADD_INITIAL_SEG = "/add_initial_seg_interaction"

# Body content type for endpoints that ship a packed numpy array.
CONTENT_TYPE_OCTET_STREAM = "application/octet-stream"
