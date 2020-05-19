; RUN: opt -rce -S < %s | FileCheck %s

%struct.copy = type { i64 }

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) nounwind

declare void @llvm.copy.start(%struct.copy* nocapture, %struct.copy* nocapture) nounwind
declare void @llvm.copy.end(%struct.copy* nocapture, %struct.copy* nocapture) nounwind

declare void @llvm.cleanup.start(%struct.copy* nocapture) nounwind
declare void @llvm.cleanup.end(%struct.copy* nocapture) nounwind

declare dso_local void @struct.copy.copyctor(%struct.copy*, %struct.copy* nonnull align 8 dereferenceable(8))
declare dso_local void @struct.copy.dtor(%struct.copy*)
declare dso_local void @modify(%struct.copy*)

; CHECK-LABEL: @test1
define void @test1() {
; CHECK-NEXT: %src = alloca %struct.copy, align 8
  %src = alloca %struct.copy, align 8
; CHECK-NOT: %dst = alloca %struct.copy, align 8
  %dst = alloca %struct.copy, align 8

; CHECK: call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %life.src)
  %life.src = bitcast %struct.copy* %src to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %life.src)

; CHECK-NOT: call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %life.dst)
  %life.dst= bitcast %struct.copy* %dst to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %life.dst)

; CHECK: %field = getelementptr inbounds %struct.copy
; CHECK-NEXT: store i64 42, i64* %field
  %field = getelementptr inbounds %struct.copy, %struct.copy* %src, i64 0, i32 0
  store i64 42, i64* %field, align 8

  call void @llvm.copy.start(%struct.copy* nonnull %dst , %struct.copy* nonnull %src)
  call void @struct.copy.copyctor(%struct.copy* nonnull %dst, %struct.copy* nonnull align 8 dereferenceable(8) %src), !copy_init !{i64 0}
  call void @llvm.copy.end(%struct.copy* nonnull %dst, %struct.copy* nonnull %src)

; CHECK-NEXT: call void @modify(%struct.copy* nonnull %src)
  call void @modify(%struct.copy* nonnull %dst)

  call void @llvm.cleanup.start(%struct.copy* nonnull %dst)
  call void @struct.copy.dtor(%struct.copy* nonnull %dst), !cleanup !{i64 0}
  call void @llvm.cleanup.end(%struct.copy* nonnull %dst)

; CHECK-NEXT: call void @llvm.cleanup.start{{.*}}(%struct.copy* nonnull %src)
; CHECK-NEXT: call void @struct.copy.dtor(%struct.copy* nonnull %src), !cleanup
; CHECK-NEXT: call void @llvm.cleanup.end{{.*}}(%struct.copy* nonnull %src)
  call void @llvm.cleanup.start(%struct.copy* nonnull %src)
  call void @struct.copy.dtor(%struct.copy* nonnull %src), !cleanup !{i64 0}
  call void @llvm.cleanup.end(%struct.copy* nonnull %src)

; CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %life.src)
; CHECK-NEXT: ret void
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %life.dst)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %life.src)
  ret void
}

; CHECK-LABEL: @test2
define void @test2() {
; CHECK-NEXT: %src = alloca %struct.copy, align 8
; CHECK-NOT: %dst = alloca %struct.copy, align 8
; CHECK-NOT: %tmp = alloca %struct.copy, align 8
  %src = alloca %struct.copy, align 8
  %dst = alloca %struct.copy, align 8
  %tmp = alloca %struct.copy, align 8

; CHECK: call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %life.src)
  %life.src = bitcast %struct.copy* %src to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %life.src)

; CHECK-NEXT: %field = getelementptr inbounds %struct.copy
; CHECK-NEXT: store i64 42, i64* %field
  %field = getelementptr inbounds %struct.copy, %struct.copy* %dst, i64 0, i32 0
  store i64 42, i64* %field, align 8

; CHECK-NOT: call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %life.dst)
  %life.dst = bitcast %struct.copy* %dst to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %life.dst)

; CHECK-NOT: call void @struct.copy.copyctor(%struct.copy* nonnull %tmp, %struct.copy* nonnull align 8 dereferenceable(8) %src), !copy_init !{i64 0}
  call void @llvm.copy.start(%struct.copy* nonnull %tmp, %struct.copy* nonnull %src)
  call void @struct.copy.copyctor(%struct.copy* nonnull %tmp, %struct.copy* nonnull align 8 dereferenceable(8) %src), !copy_init !{i64 0}
  call void @llvm.copy.end(%struct.copy* nonnull %tmp, %struct.copy* nonnull %src)

; CHECK-NOT: call void @struct.copy.copyctor(%struct.copy* nonnull %dst, %struct.copy* nonnull align 8 dereferenceable(8) %tmp)
  call void @llvm.copy.start(%struct.copy* nonnull %dst, %struct.copy* nonnull %tmp)
  call void @struct.copy.copyctor(%struct.copy* nonnull %dst, %struct.copy* nonnull align 8 dereferenceable(8) %tmp), !copy_init !{i64 0}
  call void @llvm.copy.end(%struct.copy* nonnull %dst, %struct.copy* nonnull %tmp)

; CHECK: call void @modify(%struct.copy* nonnull %src)
  call void @modify(%struct.copy* nonnull %dst)

; CHECK-NOT: call void @struct.copy.dtor(%struct.copy* nonnull %dst)
  call void @llvm.cleanup.start(%struct.copy* nonnull %dst)
  call void @struct.copy.dtor(%struct.copy* nonnull %dst), !cleanup !{i64 0}
  call void @llvm.cleanup.end(%struct.copy* nonnull %dst)

; CHECK-NOT: call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %life.dst)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %life.dst)

; CHECK-NOT: call void @struct.copy.dtor(%struct.copy* nonnull %tmp)
  call void @llvm.cleanup.start(%struct.copy* nonnull %tmp)
  call void @struct.copy.dtor(%struct.copy* nonnull %tmp), !cleanup !{i64 0}
  call void @llvm.cleanup.end(%struct.copy* nonnull %tmp)

; CHECK-NEXT: call void @llvm.cleanup.start{{.*}}(%struct.copy* nonnull %src)
; CHECK-NEXT: call void @struct.copy.dtor(%struct.copy* nonnull %src), !cleanup
; CHECK-NEXT: call void @llvm.cleanup.end{{.*}}(%struct.copy* nonnull %src)
  call void @llvm.cleanup.start(%struct.copy* nonnull %src)
  call void @struct.copy.dtor(%struct.copy* nonnull %src), !cleanup !{i64 0}
  call void @llvm.cleanup.end(%struct.copy* nonnull %src)

; CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %life.src)
; CHECK-NEXT: ret void
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %life.src)
  ret void
}
